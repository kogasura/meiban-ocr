/**
 * Sliding-window 検出器 (組込デフォルト)。
 *
 * 学習・追加モデル不要。すべての可能性のある窓を列挙して CRNN に投げる総当たり方式。
 * 速度より「ゼロ依存で動く」ことを優先。
 *
 * - 100ラベル超のフレームには遅すぎる (数百ms-数秒)
 * - Reticle 切り出し済の小さい入力 (e.g. 600x200) なら 50-150ms
 * - 全体フレームで多数ラベルを実用速度で扱いたい場合は
 *   `createOpenCvDetector` への切替を推奨
 */

import type { BBox, DetectorFn } from './types';

export interface SlidingWindowOptions {
  /** 入力画像の最大長辺 (これを超えると等比リサイズ)。デフォルト 1024。*/
  maxInputDim?: number;
  /** window 高さ (px)。デフォルト 32 (= INPUT_HEIGHT)。*/
  windowHeight?: number;
  /** window 幅 (px)。デフォルト 128 (= INPUT_WIDTH)。*/
  windowWidth?: number;
  /** 横方向の stride。デフォルト 32 (window 幅の 1/4)。*/
  strideX?: number;
  /** 縦方向の stride。デフォルト 16 (window 高さの 1/2)。*/
  strideY?: number;
  /** マルチスケール用の追加倍率。例: [1.0, 1.4, 0.7]。 */
  scales?: number[];
  /** 候補数の上限 (安全弁、stride 誤設定検知)。 */
  hardLimit?: number;
}

const DEFAULTS: Required<Omit<SlidingWindowOptions, 'hardLimit'>> & {
  hardLimit: number;
} = {
  maxInputDim: 1024,
  windowHeight: 32,
  windowWidth: 128,
  strideX: 32,
  strideY: 16,
  scales: [1.0],
  hardLimit: 20000,
};

export interface ImageSize {
  width: number;
  height: number;
}

export function computeDownscale(size: ImageSize, maxInputDim: number): number {
  const longSide = Math.max(size.width, size.height);
  return longSide > maxInputDim ? maxInputDim / longSide : 1.0;
}

export function* generateWindowBoxes(
  size: ImageSize,
  options: SlidingWindowOptions = {},
): Generator<BBox> {
  const opts = { ...DEFAULTS, ...options };
  for (const userScale of opts.scales) {
    const baseScale = computeDownscale(size, opts.maxInputDim);
    const scale = baseScale * userScale;
    const detW = Math.round(size.width * scale);
    const detH = Math.round(size.height * scale);
    if (detW < opts.windowWidth || detH < opts.windowHeight) continue;
    for (let y = 0; y <= detH - opts.windowHeight; y += opts.strideY) {
      for (let x = 0; x <= detW - opts.windowWidth; x += opts.strideX) {
        yield boxToImageSpace(x, y, opts.windowWidth, opts.windowHeight, scale);
      }
    }
    const remX = (detW - opts.windowWidth) % opts.strideX;
    const remY = (detH - opts.windowHeight) % opts.strideY;
    if (remX > 0) {
      const x = detW - opts.windowWidth;
      for (let y = 0; y <= detH - opts.windowHeight; y += opts.strideY) {
        yield boxToImageSpace(x, y, opts.windowWidth, opts.windowHeight, scale);
      }
    }
    if (remY > 0) {
      const y = detH - opts.windowHeight;
      for (let x = 0; x <= detW - opts.windowWidth; x += opts.strideX) {
        yield boxToImageSpace(x, y, opts.windowWidth, opts.windowHeight, scale);
      }
    }
  }
}

function boxToImageSpace(
  x: number,
  y: number,
  w: number,
  h: number,
  scale: number,
): BBox {
  return [
    Math.round(x / scale),
    Math.round(y / scale),
    Math.round((x + w) / scale),
    Math.round((y + h) / scale),
  ] as const;
}

export function collectWindowBoxes(
  size: ImageSize,
  options: SlidingWindowOptions = {},
): BBox[] {
  const hardLimit = options.hardLimit ?? DEFAULTS.hardLimit;
  const out: BBox[] = [];
  for (const b of generateWindowBoxes(size, options)) {
    out.push(b);
    if (out.length > hardLimit) {
      throw new Error(
        `proposal count > ${hardLimit} — stride too small or input too large`,
      );
    }
  }
  return out;
}

/**
 * 関数ファクトリ。`MeibanOCR.create({ detector: createSlidingWindowDetector({...}) })` で渡せる。
 * 引数なしで呼ぶとデフォルト設定。
 */
export function createSlidingWindowDetector(
  options: SlidingWindowOptions = {},
): DetectorFn {
  return (image: ImageData) =>
    collectWindowBoxes({ width: image.width, height: image.height }, options);
}
