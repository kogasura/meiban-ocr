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
import { detBoxBBox, detBoxQuad, type DetBox } from './detectors/types';

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
  return cropResizeGrayNormalize(
    src.data, src.width, src.height, effective,
  );
}

/**
 * RGBA バッファの指定領域を INPUT_WIDTH×INPUT_HEIGHT へリサイズし、
 * グレースケール (Rec.709) + [-1, 1] 正規化した Float32Array を返す。
 *
 * Why canvas-free: 旧実装は box ごとに canvas を2枚生成し、フレーム全体を
 * putImageData でコピーしていた (box 50-180個 × 300ms 間隔 = 毎秒数百 canvas)。
 * iOS Safari は canvas バッキングストアに厳しいページ単位の上限があり、GC が
 * 追いつかず WebContent ごと kill される (実機クラッシュの主犯)。純 JS にすると
 * 確保するのは GC 可能な TypedArray のみで、canvas 予算を一切消費しない。
 *
 * リサイズは縮小方向 = 面積平均 (cv2.INTER_AREA 相当 = 訓練前処理と同一)、
 * 拡大方向 = 線形補間の分離型2パス。旧 canvas drawImage(双線形系) より
 * 訓練分布への一致がむしろ良い。
 */
export function cropResizeGrayNormalize(
  data: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  bbox: readonly [number, number, number, number],
): Float32Array {
  // 画像境界に clip (runtime 側は canvas が自動で fill していた挙動を踏襲し、
  // はみ出しは無視して有効領域のみ使う)
  const x1 = Math.max(0, Math.min(srcW - 1, Math.floor(bbox[0])));
  const y1 = Math.max(0, Math.min(srcH - 1, Math.floor(bbox[1])));
  const x2 = Math.max(x1 + 1, Math.min(srcW, Math.ceil(bbox[2])));
  const y2 = Math.max(y1 + 1, Math.min(srcH, Math.ceil(bbox[3])));
  const cw = x2 - x1;
  const ch = y2 - y1;

  // 1) 領域を gray (Rec.709) の Float 行列へ (box 毎 churn 回避のためスクラッチ再利用)
  const gray = scratchF32(cw * ch, 0);
  for (let y = 0; y < ch; y++) {
    let si = ((y1 + y) * srcW + x1) * 4;
    let di = y * cw;
    for (let x = 0; x < cw; x++, si += 4, di++) {
      gray[di] =
        0.2126 * data[si]! + 0.7152 * data[si + 1]! + 0.0722 * data[si + 2]!;
    }
  }

  // 2) 分離型リサイズ: 横 cw→INPUT_WIDTH、縦 ch→INPUT_HEIGHT
  const horiz = resizeAxis(gray, cw, ch, INPUT_WIDTH, true, scratchF32(INPUT_WIDTH * ch, 1));
  const resized = resizeAxis(
    horiz, INPUT_WIDTH, ch, INPUT_HEIGHT, false,
    scratchF32(INPUT_WIDTH * INPUT_HEIGHT, 2),
  );

  // 3) 正規化
  const out = new Float32Array(INPUT_HEIGHT * INPUT_WIDTH);
  for (let i = 0; i < out.length; i++) {
    out[i] = (resized[i]! / 255 - NORM_MEAN) / NORM_STD;
  }
  return out;
}

/**
 * 1軸ぶんのリサイズ (horizontal=true なら幅方向、false なら高さ方向)。
 * 縮小 = 面積平均 (box filter、端は端数重み)、拡大 = 線形補間。
 */
function resizeAxis(
  src: Float32Array,
  srcW: number,
  srcH: number,
  outLen: number,
  horizontal: boolean,
  dstScratch?: Float32Array,
): Float32Array {
  const srcLen = horizontal ? srcW : srcH;
  const lines = horizontal ? srcH : srcW;
  const dst = dstScratch ?? new Float32Array(outLen * lines);
  const scale = srcLen / outLen;

  const srcAt = horizontal
    ? (line: number, i: number) => src[line * srcW + i]!
    : (line: number, i: number) => src[i * srcW + line]!;
  const dstAt = horizontal
    ? (line: number, o: number, v: number) => { dst[line * outLen + o] = v; }
    : (line: number, o: number, v: number) => { dst[o * srcW + line] = v; };

  if (scale > 1) {
    // 縮小: [o*scale, (o+1)*scale) の面積平均
    for (let o = 0; o < outLen; o++) {
      const s0 = o * scale;
      const s1 = (o + 1) * scale;
      const i0 = Math.floor(s0);
      const i1 = Math.min(srcLen, Math.ceil(s1));
      for (let line = 0; line < lines; line++) {
        let sum = 0;
        let wsum = 0;
        for (let i = i0; i < i1; i++) {
          const w = Math.min(i + 1, s1) - Math.max(i, s0);
          sum += srcAt(line, i) * w;
          wsum += w;
        }
        dstAt(line, o, sum / wsum);
      }
    }
  } else {
    // 拡大: ピクセル中心の線形補間
    for (let o = 0; o < outLen; o++) {
      const s = Math.min(srcLen - 1, Math.max(0, (o + 0.5) * scale - 0.5));
      const i0 = Math.floor(s);
      const i1 = Math.min(srcLen - 1, i0 + 1);
      const f = s - i0;
      for (let line = 0; line < lines; line++) {
        dstAt(line, o, srcAt(line, i0) * (1 - f) + srcAt(line, i1) * f);
      }
    }
  }
  return dst;
}

/**
 * 回転 quad を透視変換で水平矩形 crop に矯正し、INPUT_WIDTH×INPUT_HEIGHT に
 * リサイズ + 正規化する (PaddleOCR get_rotate_crop_image 相当)。
 *
 * Why: paddle det の quad をそのまま外接矩形 crop すると、傾いたシリアルは crop 内で
 * 斜めのまま 32×128 に潰れ、訓練分布 (水平タイト crop) から乖離する。E2E 実測で
 * rect crop 37.7% → quad 矯正 65.9% (tools/eval_end_to_end.py @960 全件)。
 *
 * recenter は適用しない (quad が既に text にタイトなため不要)。
 */
export function warpQuadAndNormalize(
  src: ImageData,
  quad: readonly [
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
  ],
): Float32Array {
  const warped = warpQuadToImage(src, quad);
  // canvas を経由せず、warp 済み RGBA を直接リサイズ+正規化 (canvas 予算を消費しない)
  return cropResizeGrayNormalize(
    warped.data, warped.width, warped.height,
    [0, 0, warped.width, warped.height],
  );
}

/**
 * 透視変換の計算本体 (canvas 非依存・テスト可能)。quad を水平矩形 RGBA に矯正する。
 * 縦長 (h/w >= 1.5) は 90° 反時計回りに回転して返す。
 */
export function warpQuadToImage(
  src: ImageData,
  quad: readonly [
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
  ],
): { data: Uint8ClampedArray; width: number; height: number } {
  const [tl, tr, br, bl] = quad;
  const dist = (a: readonly [number, number], b: readonly [number, number]) =>
    Math.hypot(a[0] - b[0], a[1] - b[1]);
  let w = Math.max(1, Math.round(Math.max(dist(tl, tr), dist(br, bl))));
  let h = Math.max(1, Math.round(Math.max(dist(tl, bl), dist(tr, br))));
  // JS per-pixel warp のコスト上限。最終入力は 128×32 なので 1024×256 (8倍) を超える
  // 解像度は精度に寄与しない。巨大 box (誤検出含む) での暴走を防ぐ。
  const MAX_W = 1024;
  const MAX_H = 256;
  if (w > MAX_W) {
    h = Math.max(1, Math.round((h * MAX_W) / w));
    w = MAX_W;
  }
  if (h > MAX_H) {
    w = Math.max(1, Math.round((w * MAX_H) / h));
    h = MAX_H;
  }

  // 単位正方形 (u,v) → quad の射影変換係数 (Heckbert)。
  // (X, Y) = ((a*u + b*v + c) / (g*u + h*v + 1), (d*u + e*v + f) / (g*u + h*v + 1))
  const sx = tl[0] - tr[0] + br[0] - bl[0];
  const sy = tl[1] - tr[1] + br[1] - bl[1];
  let g = 0;
  let hcoef = 0;
  if (Math.abs(sx) > 1e-9 || Math.abs(sy) > 1e-9) {
    const dx1 = tr[0] - br[0];
    const dx2 = bl[0] - br[0];
    const dy1 = tr[1] - br[1];
    const dy2 = bl[1] - br[1];
    const den = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(den) > 1e-9) {
      g = (sx * dy2 - dx2 * sy) / den;
      hcoef = (dx1 * sy - sx * dy1) / den;
    }
  }
  const a = tr[0] - tl[0] + g * tr[0];
  const b = bl[0] - tl[0] + hcoef * bl[0];
  const c = tl[0];
  const d = tr[1] - tl[1] + g * tr[1];
  const e = bl[1] - tl[1] + hcoef * bl[1];
  const f = tl[1];

  // 出力ピクセルごとに逆マップ + bilinear サンプル (境界は replicate)
  const sw = src.width;
  const sh = src.height;
  const sdata = src.data;
  // box 毎の churn を避けるためスクラッチを再利用 (最大 1024×256×4 = 1MB)。
  // 返り値のバッファは次の warpQuadToImage / 90°回転で上書きされるため、
  // 呼び出し側は次の呼び出し前に消費すること (現行の全呼び出し元は即時消費)。
  let warped = scratchU8(w * h * 4, 0);
  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      const denom = g * u + hcoef * v + 1;
      const sxF = (a * u + b * v + c) / denom;
      const syF = (d * u + e * v + f) / denom;
      const x0 = Math.floor(sxF - 0.5);
      const y0 = Math.floor(syF - 0.5);
      const fx = sxF - 0.5 - x0;
      const fy = syF - 0.5 - y0;
      const cx0 = Math.min(sw - 1, Math.max(0, x0));
      const cx1 = Math.min(sw - 1, Math.max(0, x0 + 1));
      const cy0 = Math.min(sh - 1, Math.max(0, y0));
      const cy1 = Math.min(sh - 1, Math.max(0, y0 + 1));
      const i00 = (cy0 * sw + cx0) * 4;
      const i01 = (cy0 * sw + cx1) * 4;
      const i10 = (cy1 * sw + cx0) * 4;
      const i11 = (cy1 * sw + cx1) * 4;
      const o = (y * w + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const top = sdata[i00 + ch]! * (1 - fx) + sdata[i01 + ch]! * fx;
        const bot = sdata[i10 + ch]! * (1 - fx) + sdata[i11 + ch]! * fx;
        warped[o + ch] = top * (1 - fy) + bot * fy;
      }
      warped[o + 3] = 255;
    }
  }

  // 縦長 crop は 90° 回転 (np.rot90 と同じ反時計回り; 縦配置テキスト対応)
  if (h / w >= 1.5) {
    const rot = scratchU8(w * h * 4, 1);
    // rot[i][j] = warped[j][w-1-i] (出力は h'=w 行 × w'=h 列)
    for (let i = 0; i < w; i++) {
      for (let j = 0; j < h; j++) {
        const so = (j * w + (w - 1 - i)) * 4;
        const do_ = (i * h + j) * 4;
        rot[do_] = warped[so]!;
        rot[do_ + 1] = warped[so + 1]!;
        rot[do_ + 2] = warped[so + 2]!;
        rot[do_ + 3] = 255;
      }
    }
    return { data: rot, width: h, height: w };
  }

  return { data: warped, width: w, height: h };
}

/**
 * 複数 box をまとめてバッチ用テンソル (N, 1, H, W) に変換。
 * Float32Array は連続メモリで [n=0 のH*W, n=1 のH*W, ...] の順。
 *
 * 要素は素の BBox (axis-aligned crop) と QuadBox (透視変換で水平矯正) の混在可。
 * `options.recenter` (default true) は BBox crop のみに適用される。
 */
export function cropAndNormalizeBatch(
  src: ImageData,
  boxes: ReadonlyArray<DetBox>,
  options: CropOptions = {},
): Float32Array {
  const stride = INPUT_HEIGHT * INPUT_WIDTH;
  const out = new Float32Array(boxes.length * stride);
  for (let n = 0; n < boxes.length; n++) {
    const quad = detBoxQuad(boxes[n]!);
    const single = quad
      ? warpQuadAndNormalize(src, quad)
      : cropAndNormalize(src, detBoxBBox(boxes[n]!), options);
    out.set(single, n * stride);
  }
  return out;
}

// ===== スクラッチプール =====
// 容量がピークに達したら以後再利用 (wasm と違い TypedArray は GC 可能だが、
// 毎フレーム数十MB の churn は iOS の GC が追いつかずピーク RSS を押し上げる)。
// slot 分けで「同一呼び出し内で2本同時に使う」ケースの衝突を防ぐ。
const f32Pool: Array<Float32Array | null> = [null, null, null];
const u8Pool: Array<Uint8ClampedArray | null> = [null, null];

function scratchF32(n: number, slot: number): Float32Array {
  const cur = f32Pool[slot];
  if (cur && cur.length >= n) return cur.subarray(0, n) as Float32Array;
  const buf = new Float32Array(n);
  f32Pool[slot] = buf;
  return buf;
}

function scratchU8(n: number, slot: number): Uint8ClampedArray {
  const cur = u8Pool[slot];
  if (cur && cur.length >= n) return cur.subarray(0, n) as Uint8ClampedArray;
  const buf = new Uint8ClampedArray(n);
  u8Pool[slot] = buf;
  return buf;
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
