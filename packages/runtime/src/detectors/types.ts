/**
 * 検出器 (text-line proposal) の共通インターフェース。
 *
 * `MeibanOCR.create({ detector: fn })` に関数を渡すと、recognize 時にその関数が
 * 入力 ImageData から候補 bbox 配列を返す責務を持つ。
 *
 * 関数型シグネチャにすることで、組込 sliding-window と利用側カスタム実装
 * (OpenCV.js、learned detector、Reticle 固定 bbox など) を等しく扱える。
 */

export type BBox = readonly [number, number, number, number]; // [x1, y1, x2, y2]

export interface DetectorFn {
  (image: ImageData): BBox[] | Promise<BBox[]>;
}
