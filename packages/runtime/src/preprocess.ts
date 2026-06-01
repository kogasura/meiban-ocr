/**
 * 画像 → CRNN 入力テンソル (Float32Array) 変換。
 *
 * - 入力: ImageInput (Canvas/OffscreenCanvas/ImageBitmap/ImageData)
 * - 出力: Float32Array, shape=(N, 1, INPUT_HEIGHT, INPUT_WIDTH), planar グレースケール、
 *   各値は [-1, 1] (mean=0.5/std=0.5 で正規化、Python側 to_model_tensor と一致)
 *
 * Why pure-Canvas2D: OpenCV.js (10MB+) を依存に持たないため。
 *
 * 2026-06-01 追加: `recenterBbox`
 *   sliding-window 窓は GT text_bbox と IoU 0.47 程度しか重ならず、窓内で text が
 *   横にズレている。fixed-head OCR は「位置 p = p 番目の文字」の暗黙契約で動くので、
 *   ズレた窓を直接渡すと位置契約が破壊され recall 0% になる。crop 直前に column
 *   activity (Sobel 風の差分) で text 重心を求め、bbox を水平シフトして再センタリング
 *   する。 augmentation で吸収しようとした v3 はアーキの位置固定と矛盾して失敗
 *   (augment_v3_position_breaking.py 参照)。
 */

import { INPUT_HEIGHT, INPUT_WIDTH, NORM_MEAN, NORM_STD } from './constants';

export type ImageInput =
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | ImageData;

export interface RecenterOptions {
  /**
   * 検索範囲を bbox 幅の何倍まで広げるか (左右それぞれ)。default 0.3。
   * text が bbox の端からはみ出ていても拾えるようにする。
   */
  expandRatio?: number;
  /**
   * 列方向 activity の平均値がこの値未満なら no-op (uniform 領域の fallback)。
   * default 5 (経験値、 0..1530 スケール: |Δgray| をピクセル数で平均)。
   */
  minActivity?: number;
  /**
   * シフト量の上限を bbox 幅の何倍までに制限するか。default 0.4 (極端なジャンプ防止)。
   */
  maxShiftRatio?: number;
}

const RECENTER_DEFAULTS: Required<RecenterOptions> = {
  expandRatio: 0.3,
  minActivity: 5,
  maxShiftRatio: 0.4,
};

/**
 * bbox 内 text を水平方向に再センタリングして新 bbox を返す。
 *
 * アルゴリズム:
 *  1. bbox を `expandRatio` ぶん左右に広げた検索範囲を取る
 *  2. 各列について `|gray(x+1, y) - gray(x-1, y)|` を縦に積算 (1D 水平勾配)
 *  3. 3 タップ平均でノイズ除去
 *  4. activity の重み付き重心を求め、それを新 bbox の中心 x にする
 *  5. activity 平均が `minActivity` 未満なら no-op (uniform 領域、 text なし)
 *  6. シフト量を `maxShiftRatio * bw` で clamp、画像境界もクランプ
 *
 * 縦方向はシフトしない (window 高さ 32 ≈ text 高さなので余裕がない)。
 */
export function recenterBbox(
  image: ImageData,
  bbox: readonly [number, number, number, number],
  options: RecenterOptions = {},
): [number, number, number, number] {
  const opts = { ...RECENTER_DEFAULTS, ...options };
  const [x1, y1, x2, y2] = bbox;
  const W = image.width;
  const H = image.height;
  const bw = x2 - x1;
  const bh = y2 - y1;
  if (bw <= 2 || bh <= 0) {
    return [x1, y1, x2, y2];
  }

  // 1) 検索範囲 (画像境界でクランプ)
  const expand = Math.round(bw * opts.expandRatio);
  const sx1 = Math.max(0, x1 - expand);
  const sx2 = Math.min(W, x2 + expand);
  const sw = sx2 - sx1;
  if (sw <= 2) {
    return [x1, y1, x2, y2];
  }
  const sy1 = Math.max(1, y1);
  const sy2 = Math.min(H - 1, y2);
  if (sy2 <= sy1) {
    return [x1, y1, x2, y2];
  }

  // 2) 列ごとの 1D 水平勾配積算
  const data = image.data;
  const colActivity = new Float32Array(sw);
  for (let y = sy1; y < sy2; y++) {
    const rowBase = y * W;
    for (let dx = 1; dx < sw - 1; dx++) {
      const x = sx1 + dx;
      const iL = (rowBase + (x - 1)) * 4;
      const iR = (rowBase + (x + 1)) * 4;
      const yL = 0.2126 * data[iL]! + 0.7152 * data[iL + 1]! + 0.0722 * data[iL + 2]!;
      const yR = 0.2126 * data[iR]! + 0.7152 * data[iR + 1]! + 0.0722 * data[iR + 2]!;
      colActivity[dx]! += yR > yL ? yR - yL : yL - yR;
    }
  }

  // 3) 3 タップ平均で smoothing (両端は 0、勾配ループが [1, sw-2] にしか書かないため)
  const smoothed = new Float32Array(sw);
  for (let i = 1; i < sw - 1; i++) {
    smoothed[i] = (colActivity[i - 1]! + colActivity[i]! + colActivity[i + 1]!) / 3;
  }

  // 4) 重心 + 5) fallback 判定
  let totalActivity = 0;
  let weightedSum = 0;
  for (let i = 0; i < sw; i++) {
    totalActivity += smoothed[i]!;
    weightedSum += i * smoothed[i]!;
  }
  // 単位: activity 合計を (sw × 実効ストリップ高 (sy2 - sy1)) で割れば 1 ピクセル平均。
  // 実際は中央列 [1, sw-2] にしか活性が入らないので 2 列ぶん 0 が混じるが、
  // 両側 (TS / Python) で同じ係数なので defaults とのキャリブレーションは一致する。
  const avgActivity = totalActivity / Math.max(1, sw * (sy2 - sy1));
  if (avgActivity < opts.minActivity || totalActivity <= 0) {
    return [x1, y1, x2, y2];
  }

  const centroidLocal = weightedSum / totalActivity; // 0..sw 内の重心
  const centroidImageX = sx1 + centroidLocal; // 画像座標系の重心 x

  // 6) シフト量 clamp + 境界 clamp
  const oldCenterX = (x1 + x2) / 2;
  let dx = centroidImageX - oldCenterX;
  const maxShift = opts.maxShiftRatio * bw;
  if (dx > maxShift) dx = maxShift;
  if (dx < -maxShift) dx = -maxShift;

  let newX1 = Math.round(x1 + dx);
  if (newX1 < 0) newX1 = 0;
  if (newX1 + bw > W) newX1 = W - bw;
  return [newX1, y1, newX1 + bw, y2];
}

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

export interface CropOptions {
  /**
   * crop 前に bbox を text 中央に寄せ直すか (Phase 2b, 2026-06-01)。
   * - `true` (default) / option: recenterBbox 適用
   * - `false`: 旧挙動 (bbox をそのまま使用)
   */
  recenter?: boolean | RecenterOptions;
}

/**
 * 指定領域を bbox で切り出し、INPUT_HEIGHTxINPUT_WIDTH にリサイズし、
 * グレースケール + 正規化した planar Float32Array を返す (shape=(1, H, W) 相当の連続配列)。
 *
 * `options.recenter` が真値なら、crop 前に `recenterBbox` で水平方向のズレを
 * 補正する。fixed-head の位置固定契約 (位置 p = p 番目の文字) を保つために必要。
 */
export function cropAndNormalize(
  src: ImageData,
  bbox: readonly [number, number, number, number],
  options: CropOptions = {},
): Float32Array {
  const rc = options.recenter ?? true;
  const effective: readonly [number, number, number, number] = rc === false
    ? bbox
    : recenterBbox(src, bbox, typeof rc === 'object' ? rc : {});
  const [x1, y1, x2, y2] = effective;
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
 *
 * `options.recenter` (default true) で `cropAndNormalize` 側の再センタリングを制御。
 */
export function cropAndNormalizeBatch(
  src: ImageData,
  bboxes: ReadonlyArray<readonly [number, number, number, number]>,
  options: CropOptions = {},
): Float32Array {
  const stride = INPUT_HEIGHT * INPUT_WIDTH;
  const out = new Float32Array(bboxes.length * stride);
  for (let n = 0; n < bboxes.length; n++) {
    const single = cropAndNormalize(src, bboxes[n]!, options);
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
