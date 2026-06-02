/**
 * DB (Differentiable Binarization) postprocess — PP-OCRv4 det 出力の segmentation map
 * から text bbox を抽出する。
 *
 * 簡略版実装:
 *   1. segmentation map (sigmoid output) を thresh で 二値化
 *   2. 8-連結成分ラベリングで blob を取得
 *   3. 各 blob を axis-aligned rect で囲む (PaddleOCR の polygon refinement は省略)
 *   4. score = blob 内の seg map 平均が score_threshold を超えたものだけ残す
 *   5. unclip: bbox を中心 outward に ratio 倍で expand
 *
 * 入力 seg map shape: (1, 1, segH, segW)、 通常 segH = origH/4, segW = origW/4
 * 出力 bbox は **元画像座標系の axis-aligned rect** [x1, y1, x2, y2]。
 *
 * 参考: PaddleOCR Python 実装 (rapidocr_onnxruntime/ch_ppocr_det/utils.py)。
 * 軸並行 rect で簡略化しているため、 斜め配置の text は精度がやや落ちる。
 * 銘板用途は基本軸並行なので影響軽微。
 */

export interface DBOptions {
  /** 二値化閾値。 default 0.3 (PaddleOCR config と同じ)。 */
  binaryThreshold?: number;
  /** blob 採択 score 閾値。 default 0.5。 */
  scoreThreshold?: number;
  /** 検出 bbox の最小幅/高さ (px、 seg map 解像度上)。 default 3。 */
  minBoxSize?: number;
  /** unclip 倍率。 default 1.6 (PaddleOCR config と同じ)。 */
  unclipRatio?: number;
  /** seg map → 元画像のスケール比 (default は seg_h / orig_h を渡す)。 */
  scaleX?: number;
  scaleY?: number;
}

const DEFAULTS: Required<Omit<DBOptions, 'scaleX' | 'scaleY'>> = {
  binaryThreshold: 0.3,
  scoreThreshold: 0.5,
  minBoxSize: 3,
  unclipRatio: 1.6,
};

/**
 * Segmentation map → bbox 抽出。
 *
 * @param segMap Float32Array length = segH * segW, sigmoid 確率
 * @param segH segmentation map 高さ
 * @param segW segmentation map 幅
 * @param origH 元画像高さ (出力 bbox は orig 座標)
 * @param origW 元画像幅
 * @param options
 */
export function dbPostprocess(
  segMap: Float32Array,
  segH: number,
  segW: number,
  origH: number,
  origW: number,
  options: DBOptions = {},
): Array<[number, number, number, number]> {
  const opts = { ...DEFAULTS, ...options };

  // 1. 二値化
  const binary = new Uint8Array(segH * segW);
  for (let i = 0; i < segMap.length; i++) {
    binary[i] = segMap[i]! >= opts.binaryThreshold ? 1 : 0;
  }

  // 2. 8-連結成分ラベリング (2-pass)
  const labels = new Int32Array(segH * segW);
  const equiv: Map<number, number> = new Map();
  let nextLabel = 1;

  // 1st pass: ラベル付与 + 等価ペア記録
  for (let y = 0; y < segH; y++) {
    for (let x = 0; x < segW; x++) {
      const i = y * segW + x;
      if (binary[i] === 0) continue;
      const neighborLabels: number[] = [];
      // 左、 上、 左上、 右上の 4 つを見る (8-連結で過去処理済みの近傍)
      if (x > 0 && labels[i - 1]! > 0) neighborLabels.push(labels[i - 1]!);
      if (y > 0 && labels[i - segW]! > 0) neighborLabels.push(labels[i - segW]!);
      if (x > 0 && y > 0 && labels[i - segW - 1]! > 0)
        neighborLabels.push(labels[i - segW - 1]!);
      if (x < segW - 1 && y > 0 && labels[i - segW + 1]! > 0)
        neighborLabels.push(labels[i - segW + 1]!);

      if (neighborLabels.length === 0) {
        labels[i] = nextLabel++;
      } else {
        const minLabel = Math.min(...neighborLabels);
        labels[i] = minLabel;
        for (const nl of neighborLabels) {
          if (nl !== minLabel) {
            unionLabels(equiv, nl, minLabel);
          }
        }
      }
    }
  }

  // 2nd pass: equiv を解決して final label に統一
  const finalLabel = (l: number): number => {
    let cur = l;
    while (equiv.has(cur)) {
      const next = equiv.get(cur)!;
      if (next === cur) break;
      cur = next;
    }
    return cur;
  };

  // 各 final label の bbox + score 集計
  type BlobStat = {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    scoreSum: number;
    pixelCount: number;
  };
  const blobs: Map<number, BlobStat> = new Map();
  for (let y = 0; y < segH; y++) {
    for (let x = 0; x < segW; x++) {
      const i = y * segW + x;
      const l = labels[i]!;
      if (l === 0) continue;
      const fl = finalLabel(l);
      let stat = blobs.get(fl);
      if (!stat) {
        stat = {
          minX: x,
          maxX: x,
          minY: y,
          maxY: y,
          scoreSum: 0,
          pixelCount: 0,
        };
        blobs.set(fl, stat);
      } else {
        if (x < stat.minX) stat.minX = x;
        if (x > stat.maxX) stat.maxX = x;
        if (y < stat.minY) stat.minY = y;
        if (y > stat.maxY) stat.maxY = y;
      }
      stat.scoreSum += segMap[i]!;
      stat.pixelCount += 1;
    }
  }

  // 3. score gate + サイズ gate + unclip + orig 座標変換
  const scaleX = opts.scaleX ?? origW / segW;
  const scaleY = opts.scaleY ?? origH / segH;
  const out: Array<[number, number, number, number]> = [];

  for (const stat of blobs.values()) {
    const score = stat.scoreSum / Math.max(1, stat.pixelCount);
    if (score < opts.scoreThreshold) continue;

    const w = stat.maxX - stat.minX + 1;
    const h = stat.maxY - stat.minY + 1;
    if (w < opts.minBoxSize || h < opts.minBoxSize) continue;

    // unclip: 中心 outward に ratio 倍 expand
    // bbox は半開区間 [minX, maxX+1) として center 計算 (= cell minX..maxX を完全に含む)
    const cx = (stat.minX + stat.maxX + 1) / 2;
    const cy = (stat.minY + stat.maxY + 1) / 2;
    const halfW = (w * opts.unclipRatio) / 2;
    const halfH = (h * opts.unclipRatio) / 2;

    let x1 = Math.round((cx - halfW) * scaleX);
    let y1 = Math.round((cy - halfH) * scaleY);
    let x2 = Math.round((cx + halfW) * scaleX);
    let y2 = Math.round((cy + halfH) * scaleY);

    // クランプ
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > origW) x2 = origW;
    if (y2 > origH) y2 = origH;
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;

    out.push([x1, y1, x2, y2]);
  }

  return out;
}

function unionLabels(equiv: Map<number, number>, a: number, b: number): void {
  const rootA = findRoot(equiv, a);
  const rootB = findRoot(equiv, b);
  if (rootA === rootB) return;
  const minRoot = Math.min(rootA, rootB);
  const maxRoot = Math.max(rootA, rootB);
  equiv.set(maxRoot, minRoot);
}

function findRoot(equiv: Map<number, number>, l: number): number {
  let cur = l;
  while (equiv.has(cur)) {
    const next = equiv.get(cur)!;
    if (next === cur) break;
    cur = next;
  }
  return cur;
}
