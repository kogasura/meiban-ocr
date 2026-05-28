/**
 * 画像 → CRNN 入力テンソル (Float32Array) 変換。
 *
 * - 入力: ImageInput (Canvas/OffscreenCanvas/ImageBitmap/ImageData)
 * - 出力: Float32Array, shape=(N, 1, INPUT_HEIGHT, INPUT_WIDTH), planar グレースケール、
 *   各値は [-1, 1] (mean=0.5/std=0.5 で正規化、Python側 to_model_tensor と一致)
 *
 * Why pure-Canvas2D: OpenCV.js (10MB+) を依存に持たないため。
 */

import { INPUT_HEIGHT, INPUT_WIDTH, NORM_MEAN, NORM_STD } from './constants';

export type ImageInput =
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | ImageData;

/** 入力を 2D context に描画し、ImageData を取り出す。 */
export function imageInputToImageData(image: ImageInput): ImageData {
  if (image instanceof ImageData) {
    return image;
  }
  if (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas) {
    const ctx = image.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context from OffscreenCanvas');
    return ctx.getImageData(0, 0, image.width, image.height);
  }
  if (image instanceof HTMLCanvasElement) {
    const ctx = image.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context from HTMLCanvasElement');
    return ctx.getImageData(0, 0, image.width, image.height);
  }
  // ImageBitmap: draw to a fresh canvas
  const canvas = makeCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('Failed to get 2D context');
  ctx.drawImage(image as unknown as CanvasImageSource, 0, 0);
  return ctx.getImageData(0, 0, image.width, image.height);
}

/**
 * 指定領域を bbox で切り出し、INPUT_HEIGHTxINPUT_WIDTH にリサイズし、
 * グレースケール + 正規化した planar Float32Array を返す (shape=(1, H, W) 相当の連続配列)。
 */
export function cropAndNormalize(
  src: ImageData,
  bbox: readonly [number, number, number, number],
): Float32Array {
  const [x1, y1, x2, y2] = bbox;
  const cw = Math.max(1, x2 - x1);
  const ch = Math.max(1, y2 - y1);

  // 1) bbox 領域を中間 canvas に描画
  const tmp = makeCanvas(cw, ch);
  const tctx = tmp.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!tctx) throw new Error('Failed to get 2D context for crop canvas');
  // putImageData は dirty rect で部分コピー可能
  tctx.putImageData(src, -x1, -y1);

  // 2) INPUT_WIDTHxINPUT_HEIGHT にリサイズ
  const resized = makeCanvas(INPUT_WIDTH, INPUT_HEIGHT);
  const rctx = resized.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!rctx) throw new Error('Failed to get 2D context for resize canvas');
  rctx.imageSmoothingEnabled = true;
  // OffscreenCanvas には imageSmoothingQuality があるが、HTMLCanvas にも存在する
  (rctx as CanvasRenderingContext2D).imageSmoothingQuality = 'high';
  rctx.drawImage(tmp as unknown as CanvasImageSource, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const data = rctx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT).data;

  // 3) RGB → グレースケール (Rec.709 輝度) + 正規化
  const out = new Float32Array(INPUT_HEIGHT * INPUT_WIDTH);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    // luminance 線形近似 (Rec.709 係数を Y' に近似)
    const r = data[i]!;
    const g = data[i + 1]!;
    const b = data[i + 2]!;
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    out[j] = (y / 255 - NORM_MEAN) / NORM_STD;
  }
  return out;
}

/**
 * 複数 bbox をまとめてバッチ用テンソル (N, 1, H, W) に変換。
 * Float32Array は連続メモリで [n=0 のH*W, n=1 のH*W, ...] の順。
 */
export function cropAndNormalizeBatch(
  src: ImageData,
  bboxes: ReadonlyArray<readonly [number, number, number, number]>,
): Float32Array {
  const stride = INPUT_HEIGHT * INPUT_WIDTH;
  const out = new Float32Array(bboxes.length * stride);
  for (let n = 0; n < bboxes.length; n++) {
    const single = cropAndNormalize(src, bboxes[n]!);
    out.set(single, n * stride);
  }
  return out;
}

function makeCanvas(w: number, h: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(w, h);
  }
  if (typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }
  throw new Error('Neither OffscreenCanvas nor document is available');
}
