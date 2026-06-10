/**
 * DB (Differentiable Binarization) postprocess — PP-OCRv4 det 出力の segmentation map
 * から text box を抽出する。
 *
 * 2 つの実装を提供する:
 *
 * `dbPostprocess` (簡略版・後方互換):
 *   1. segmentation map (sigmoid output) を thresh で 二値化
 *   2. 8-連結成分ラベリングで blob を取得
 *   3. 各 blob を axis-aligned rect で囲む
 *   4. score = blob 内の seg map 平均が score_threshold を超えたものだけ残す
 *   5. unclip: bbox を中心 outward に ratio 倍で expand
 *   出力は元画像座標系の axis-aligned rect [x1, y1, x2, y2]。
 *   斜め配置の text は背景を抱き込み、crop 内で文字が斜めのまま潰れるため精度が落ちる。
 *
 * `dbPostprocessQuad` (本家 PaddleOCR 準拠):
 *   3'. 各 blob を minAreaRect (回転最小外接矩形) で囲む
 *   5'. unclip: Vatti offset (d = area * ratio / perimeter) を矩形に閉形式で適用
 *   出力は回転 quad + 外接 bbox の QuadBox。crop 側で透視変換して水平矯正すると
 *   訓練分布 (水平タイト crop) に一致する。
 *   E2E 実測 (tools/eval_end_to_end.py, 同一300枚 @960): rect 46.6% → quad 65.0%
 *   (+18.4pt)、偽発火 -34%。全件 @960: 37.7% → 65.9%。
 *
 * 入力 seg map shape: (1, 1, segH, segW)、 通常 segH = origH/4, segW = origW/4
 *
 * 参考: PaddleOCR Python 実装 (rapidocr_onnxruntime/ch_ppocr_det/utils.py)。
 */

import type { BBox, Quad, QuadBox } from '../../detectors/types';

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

/** 8-連結成分 1 blob の統計。boundary は行ごとの (minX, maxX) — convex hull 計算用。 */
interface BlobStat {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  scoreSum: number;
  pixelCount: number;
  /** 行 y → その行の minX。 */
  rowMinX: Map<number, number>;
  /** 行 y → その行の maxX。 */
  rowMaxX: Map<number, number>;
}

/**
 * 二値化 + 8-連結成分ラベリング (2-pass) で blob 統計を集める。
 * dbPostprocess / dbPostprocessQuad の共通前段。
 */
function collectBlobs(
  segMap: Float32Array,
  segH: number,
  segW: number,
  binaryThreshold: number,
): BlobStat[] {
  // 1. 二値化
  const binary = new Uint8Array(segH * segW);
  for (let i = 0; i < segMap.length; i++) {
    binary[i] = segMap[i]! >= binaryThreshold ? 1 : 0;
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

  // 2nd pass: equiv を解決して final label に統一しつつ統計集計
  const blobs: Map<number, BlobStat> = new Map();
  for (let y = 0; y < segH; y++) {
    for (let x = 0; x < segW; x++) {
      const i = y * segW + x;
      const l = labels[i]!;
      if (l === 0) continue;
      const fl = findRoot(equiv, l);
      let stat = blobs.get(fl);
      if (!stat) {
        stat = {
          minX: x,
          maxX: x,
          minY: y,
          maxY: y,
          scoreSum: 0,
          pixelCount: 0,
          rowMinX: new Map(),
          rowMaxX: new Map(),
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
      const rMin = stat.rowMinX.get(y);
      if (rMin === undefined || x < rMin) stat.rowMinX.set(y, x);
      const rMax = stat.rowMaxX.get(y);
      if (rMax === undefined || x > rMax) stat.rowMaxX.set(y, x);
    }
  }

  return Array.from(blobs.values());
}

/**
 * Segmentation map → axis-aligned bbox 抽出 (簡略版・後方互換)。
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
  const scaleX = opts.scaleX ?? origW / segW;
  const scaleY = opts.scaleY ?? origH / segH;
  const out: Array<[number, number, number, number]> = [];

  for (const stat of collectBlobs(segMap, segH, segW, opts.binaryThreshold)) {
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

/**
 * Segmentation map → 回転 quad 抽出 (本家 PaddleOCR DBPostProcess 準拠)。
 *
 * blob ごとに minAreaRect → Vatti offset unclip (d = area*ratio/perimeter、 矩形なら
 * 全辺 +d と等価) → 4 角を元画像座標へ。crop 側は quad を透視変換して水平矯正する。
 */
export function dbPostprocessQuad(
  segMap: Float32Array,
  segH: number,
  segW: number,
  origH: number,
  origW: number,
  options: DBOptions = {},
): QuadBox[] {
  const opts = { ...DEFAULTS, ...options };
  const scaleX = opts.scaleX ?? origW / segW;
  const scaleY = opts.scaleY ?? origH / segH;
  const out: QuadBox[] = [];

  for (const stat of collectBlobs(segMap, segH, segW, opts.binaryThreshold)) {
    const score = stat.scoreSum / Math.max(1, stat.pixelCount);
    if (score < opts.scoreThreshold) continue;

    // 行ごとの (minX, maxX) だけで convex hull は厳密 (hull 頂点は各行の極値)
    const pts: Array<[number, number]> = [];
    for (const [y, x] of stat.rowMinX) pts.push([x, y]);
    for (const [y, x] of stat.rowMaxX) pts.push([x, y]);
    const rect = minAreaRect(pts);
    if (Math.min(rect.w, rect.h) < opts.minBoxSize) continue;

    // unclip: DB の seg は shrink 領域なので offset で復元。
    // 矩形の Vatti offset は全辺 +d (d = area * ratio / perimeter)
    const d = (rect.w * rect.h * opts.unclipRatio) / (2 * (rect.w + rect.h));
    const hw = rect.w / 2 + d;
    const hh = rect.h / 2 + d;
    // rect 軸: u = (ux, uy) が幅方向、 n = (-uy, ux) が高さ方向
    const corners: Array<[number, number]> = [
      [rect.cx - rect.ux * hw + rect.uy * hh, rect.cy - rect.uy * hw - rect.ux * hh],
      [rect.cx + rect.ux * hw + rect.uy * hh, rect.cy + rect.uy * hw - rect.ux * hh],
      [rect.cx + rect.ux * hw - rect.uy * hh, rect.cy + rect.uy * hw + rect.ux * hh],
      [rect.cx - rect.ux * hw - rect.uy * hh, rect.cy - rect.uy * hw + rect.ux * hh],
    ];
    // 元画像座標へ (異方スケールで平行四辺形になり得るが透視変換は任意 quad を扱える)
    for (const c of corners) {
      c[0] = Math.min(origW - 1, Math.max(0, c[0] * scaleX));
      c[1] = Math.min(origH - 1, Math.max(0, c[1] * scaleY));
    }
    const quad = orderQuad(corners);
    const xs = [quad[0][0], quad[1][0], quad[2][0], quad[3][0]];
    const ys = [quad[0][1], quad[1][1], quad[2][1], quad[3][1]];
    const bbox: BBox = [
      Math.floor(Math.min(...xs)),
      Math.floor(Math.min(...ys)),
      Math.ceil(Math.max(...xs)),
      Math.ceil(Math.max(...ys)),
    ];
    if (bbox[2] - bbox[0] < 1 || bbox[3] - bbox[1] < 1) continue;
    out.push({ bbox, quad });
  }

  return out;
}

/** 4点を tl, tr, br, bl 順に並べる (Python 側 _order_quad と同一規則)。 */
function orderQuad(pts: Array<[number, number]>): Quad {
  let tl = 0, tr = 0, br = 0, bl = 0;
  let minSum = Infinity, maxSum = -Infinity, minDiff = Infinity, maxDiff = -Infinity;
  for (let i = 0; i < 4; i++) {
    const s = pts[i]![0] + pts[i]![1];
    const d = pts[i]![0] - pts[i]![1];
    if (s < minSum) { minSum = s; tl = i; }
    if (s > maxSum) { maxSum = s; br = i; }
    if (d > maxDiff) { maxDiff = d; tr = i; }
    if (d < minDiff) { minDiff = d; bl = i; }
  }
  return [
    [pts[tl]![0], pts[tl]![1]],
    [pts[tr]![0], pts[tr]![1]],
    [pts[br]![0], pts[br]![1]],
    [pts[bl]![0], pts[bl]![1]],
  ];
}

interface MinAreaRect {
  cx: number;
  cy: number;
  /** 幅 (u 軸方向の長さ)。 */
  w: number;
  /** 高さ (n 軸方向の長さ)。 */
  h: number;
  /** u 軸の単位ベクトル。 */
  ux: number;
  uy: number;
}

/** 回転最小外接矩形 (rotating calipers)。cv2.minAreaRect 相当。 */
function minAreaRect(pts: Array<[number, number]>): MinAreaRect {
  const hull = convexHull(pts);
  if (hull.length === 1) {
    return { cx: hull[0]![0], cy: hull[0]![1], w: 0, h: 0, ux: 1, uy: 0 };
  }
  let best: MinAreaRect | null = null;
  let bestArea = Infinity;
  for (let i = 0; i < hull.length; i++) {
    const j = (i + 1) % hull.length;
    let ux = hull[j]![0] - hull[i]![0];
    let uy = hull[j]![1] - hull[i]![1];
    const len = Math.hypot(ux, uy);
    if (len === 0) continue;
    ux /= len;
    uy /= len;
    // 法線 n = (-uy, ux)
    let minU = Infinity, maxU = -Infinity, minN = Infinity, maxN = -Infinity;
    for (const [x, y] of hull) {
      const u = x * ux + y * uy;
      const n = -x * uy + y * ux;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (n < minN) minN = n;
      if (n > maxN) maxN = n;
    }
    const area = (maxU - minU) * (maxN - minN);
    if (area < bestArea) {
      bestArea = area;
      const cu = (minU + maxU) / 2;
      const cn = (minN + maxN) / 2;
      best = {
        cx: cu * ux - cn * uy,
        cy: cu * uy + cn * ux,
        w: maxU - minU,
        h: maxN - minN,
        ux,
        uy,
      };
    }
  }
  // hull が全点同一座標などの degenerate (len===0 のみ) fallback
  return best ?? { cx: hull[0]![0], cy: hull[0]![1], w: 0, h: 0, ux: 1, uy: 0 };
}

/** Andrew's monotone chain。返り値は反時計回り (y 下向き座標系では時計回り)。 */
function convexHull(pts: Array<[number, number]>): Array<[number, number]> {
  const sorted = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  // 重複除去
  const uniq: Array<[number, number]> = [];
  for (const p of sorted) {
    const last = uniq[uniq.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) uniq.push(p);
  }
  if (uniq.length <= 2) return uniq;
  const cross = (
    o: [number, number],
    a: [number, number],
    b: [number, number],
  ): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of uniq) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0
    )
      lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = uniq.length - 1; i >= 0; i--) {
    const p = uniq[i]!;
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0
    )
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
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
