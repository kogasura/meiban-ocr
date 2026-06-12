/**
 * PaddleOCR (PP-OCRv4) 用の画像前処理。
 *
 * Detection 用 と Recognition 用の 2 種類が必要:
 *
 * **Detection (det)**:
 *   - 入力画像をアスペクト保持で long side ≤ limit_side_len (default 960) にリサイズ
 *   - 高さ・幅は 32 の倍数に揃える (model stride 制約)
 *   - RGB → (pixel/255 - mean) / std (mean=std=[0.5,0.5,0.5])
 *   - CHW float32 にレイアウト
 *
 * **Recognition (rec)** (per bbox):
 *   - bbox 内をアフィン補正 (今回は axis-aligned rect なので単純 crop)
 *   - 高さを 48 に揃え、アスペクト保持で幅を決定 (最大 320)
 *   - 不足幅は黒 padding
 *   - 同じ正規化 + CHW
 */

// E2E実測(serial-disjoint held-out, tools/eval_end_to_end.py)で 736→960 で検出カバー
// 62→89%、e2e EM 21.8→37.7% と判明したため既定を 960 に引き上げ(2026-06-09)。
// 小シリアル(<35px)が 736 のダウンスケールで検出マップから消えるのが主因。det推論コスト
// は +24%(長辺^2)で KGI ≤300ms/frame に収まる。1280 は 4.7× で過剰。detLongSide で上書き可。
const DET_DEFAULT_LIMIT = 960;
const REC_TARGET_H = 48;
const REC_MAX_W = 320;

// preprocessForDet 用スクラッチ (同一解像度の連続フレームで再利用)
let detScratch: Float32Array | null = null;

export interface DetPreprocessResult {
  /** Float32 RGB CHW, shape [1, 3, H, W] (H, W は 32 倍数化済) */
  tensor: Float32Array;
  /** 実際にモデルに渡される H */
  height: number;
  /** 実際にモデルに渡される W */
  width: number;
  /** 元画像 → モデル入力へのリサイズ比 (model_x / orig_x = scale) */
  scaleX: number;
  scaleY: number;
}

/**
 * Detection 用画像前処理。
 */
export function preprocessForDet(
  image: ImageData,
  limitSideLen: number = DET_DEFAULT_LIMIT,
): DetPreprocessResult {
  const { width: origW, height: origH, data } = image;

  // 長辺を limitSideLen 以下にする scale を計算
  const longSide = Math.max(origW, origH);
  let scale = longSide > limitSideLen ? limitSideLen / longSide : 1.0;
  // 短辺が極端に小さくならないようガード (PaddleOCR の挙動を模倣)
  // 32 倍数に向けて切り捨て
  let newH = Math.max(32, Math.round((origH * scale) / 32) * 32);
  let newW = Math.max(32, Math.round((origW * scale) / 32) * 32);

  const scaleX = newW / origW;
  const scaleY = newH / origH;

  // カメラ解像度は一定なのでスクラッチを再利用 (毎フレーム ~11MB@1280 の churn を排除)。
  // 注意: ort.env.wasm.proxy=true だと入力バッファが transfer され detach するため、
  // detach 検出時は再確保する。
  const need = 3 * newH * newW;
  if (
    detScratch === null ||
    detScratch.length !== need ||
    detScratch.buffer.byteLength === 0
  ) {
    detScratch = new Float32Array(need);
  }
  const tensor = detScratch;

  // 双線形補間でリサイズ + 正規化を一度に行う (Float32 → CHW)。
  for (let dy = 0; dy < newH; dy++) {
    const srcY = dy / scaleY;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(origH - 1, y0 + 1);
    const wy = srcY - y0;
    for (let dx = 0; dx < newW; dx++) {
      const srcX = dx / scaleX;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(origW - 1, x0 + 1);
      const wx = srcX - x0;

      // 4 近傍を取得 (RGBA → RGB のみ使用)
      const i00 = (y0 * origW + x0) * 4;
      const i10 = (y0 * origW + x1) * 4;
      const i01 = (y1 * origW + x0) * 4;
      const i11 = (y1 * origW + x1) * 4;
      for (let c = 0; c < 3; c++) {
        const v00 = data[i00 + c]!;
        const v10 = data[i10 + c]!;
        const v01 = data[i01 + c]!;
        const v11 = data[i11 + c]!;
        const top = v00 * (1 - wx) + v10 * wx;
        const bot = v01 * (1 - wx) + v11 * wx;
        const v = top * (1 - wy) + bot * wy;
        // 正規化: (v/255 - 0.5) / 0.5  =  (v - 127.5) / 127.5
        const norm = (v - 127.5) / 127.5;
        // CHW: tensor[c, dy, dx]
        const tensorIdx = c * newH * newW + dy * newW + dx;
        tensor[tensorIdx] = norm;
      }
    }
  }

  return { tensor, height: newH, width: newW, scaleX, scaleY };
}

export interface RecPreprocessResult {
  /** Float32 RGB CHW, shape [1, 3, 48, W] (W は max 320 まで) */
  tensor: Float32Array;
  height: number;
  width: number;
}

/**
 * Recognition 用画像前処理 (1 bbox 分)。
 *
 * bbox は axis-aligned [x1, y1, x2, y2]。 PaddleOCR の幅は固定 320 で padding。
 */
export function preprocessForRec(
  image: ImageData,
  bbox: readonly [number, number, number, number],
  targetH: number = REC_TARGET_H,
  maxW: number = REC_MAX_W,
): RecPreprocessResult {
  const { width: origW, height: origH, data } = image;
  const [x1, y1, x2, y2] = bbox;
  const bw = Math.max(1, Math.min(origW, x2) - Math.max(0, x1));
  const bh = Math.max(1, Math.min(origH, y2) - Math.max(0, y1));

  // アスペクト保持で targetH に揃え、 幅を計算
  const aspectRatio = bw / bh;
  let newW = Math.ceil(targetH * aspectRatio);
  if (newW > maxW) newW = maxW;

  // tensor は [3, targetH, maxW] (padding 部分は 0)
  const tensor = new Float32Array(3 * targetH * maxW);
  // 0 で初期化済、 padding はそのまま (PaddleOCR は 0 で pad)
  // ただし「正規化後の 0」を pad に使うのが正確。 (0 - 127.5) / 127.5 = -1
  const padValue = -1.0;
  for (let i = 0; i < tensor.length; i++) {
    tensor[i] = padValue;
  }

  const sx1 = Math.max(0, x1);
  const sy1 = Math.max(0, y1);
  // 双線形補間で bbox 内を targetH × newW にリサイズ
  for (let dy = 0; dy < targetH; dy++) {
    const srcY = sy1 + (dy / targetH) * bh;
    const y0 = Math.floor(srcY);
    const y1f = Math.min(origH - 1, y0 + 1);
    const wy = srcY - y0;
    for (let dx = 0; dx < newW; dx++) {
      const srcX = sx1 + (dx / newW) * bw;
      const x0 = Math.floor(srcX);
      const x1f = Math.min(origW - 1, x0 + 1);
      const wx = srcX - x0;

      const i00 = (y0 * origW + x0) * 4;
      const i10 = (y0 * origW + x1f) * 4;
      const i01 = (y1f * origW + x0) * 4;
      const i11 = (y1f * origW + x1f) * 4;
      for (let c = 0; c < 3; c++) {
        const v00 = data[i00 + c]!;
        const v10 = data[i10 + c]!;
        const v01 = data[i01 + c]!;
        const v11 = data[i11 + c]!;
        const top = v00 * (1 - wx) + v10 * wx;
        const bot = v01 * (1 - wx) + v11 * wx;
        const v = top * (1 - wy) + bot * wy;
        const norm = (v - 127.5) / 127.5;
        const tensorIdx = c * targetH * maxW + dy * maxW + dx;
        tensor[tensorIdx] = norm;
      }
    }
  }

  return { tensor, height: targetH, width: maxW };
}

/**
 * 複数 bbox を batch tensor に統合。
 * 戻り値の tensor shape は [N, 3, 48, 320]。
 */
export function preprocessForRecBatch(
  image: ImageData,
  bboxes: ReadonlyArray<readonly [number, number, number, number]>,
  targetH: number = REC_TARGET_H,
  maxW: number = REC_MAX_W,
): { tensor: Float32Array; batch: number; height: number; width: number } {
  const sampleStride = 3 * targetH * maxW;
  const tensor = new Float32Array(bboxes.length * sampleStride);
  for (let i = 0; i < bboxes.length; i++) {
    const single = preprocessForRec(image, bboxes[i]!, targetH, maxW);
    tensor.set(single.tensor, i * sampleStride);
  }
  return { tensor, batch: bboxes.length, height: targetH, width: maxW };
}
