/**
 * 画像入力 (Canvas/OffscreenCanvas/ImageBitmap/ImageData) → ImageData 変換。
 *
 * Why pure-Canvas2D: OpenCV.js (10MB+) を依存に持たないため。
 *
 * crop / resize / 透視変換などの認識前処理は各 backend 内に閉じる
 * (paddle backend は `backends/paddle/preprocess.ts`)。ここは入力正規化のみ。
 */

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
  // ImageBitmap: 変換用 canvas は1枚をモジュールで使い回す。
  // 毎フレーム新規 canvas を作ると iOS Safari の canvas backing store 予算を
  // 食い潰す (crop 経路の canvas 全廃と同じ理由)。サイズ変更は width/height
  // 再設定で backing store を再利用する。
  if (
    bitmapCanvas === null ||
    bitmapCanvas.width !== image.width ||
    bitmapCanvas.height !== image.height
  ) {
    if (bitmapCanvas === null) {
      bitmapCanvas = makeCanvas(image.width, image.height);
    } else {
      bitmapCanvas.width = image.width;
      bitmapCanvas.height = image.height;
    }
  }
  const ctx = bitmapCanvas.getContext('2d', { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('Failed to get 2D context');
  ctx.drawImage(image as unknown as CanvasImageSource, 0, 0);
  return ctx.getImageData(0, 0, image.width, image.height);
}

// ImageBitmap → ImageData 変換用の使い回し canvas
let bitmapCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;

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
