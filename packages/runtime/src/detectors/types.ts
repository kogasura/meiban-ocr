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

/** 回転 quad (tl, tr, br, bl 順、元画像座標)。 */
export type Quad = readonly [
  readonly [number, number],
  readonly [number, number],
  readonly [number, number],
  readonly [number, number],
];

/**
 * 回転 quad つき検出 box。`bbox` は quad の axis-aligned 外接矩形
 * (NMS / prefilter / 被覆判定はこちらを使う)。crop は quad の透視変換で行う。
 */
export interface QuadBox {
  readonly bbox: BBox;
  readonly quad: Quad;
}

/** 検出器の出力要素。素の BBox (後方互換) か、回転 quad つき box。 */
export type DetBox = BBox | QuadBox;

/** DetBox から axis-aligned bbox を取り出す。 */
export function detBoxBBox(d: DetBox): BBox {
  return Array.isArray(d) ? (d as BBox) : (d as QuadBox).bbox;
}

/** DetBox から quad を取り出す (素の BBox なら undefined)。 */
export function detBoxQuad(d: DetBox): Quad | undefined {
  return Array.isArray(d) ? undefined : (d as QuadBox).quad;
}

export interface DetectorFn {
  (image: ImageData): readonly DetBox[] | Promise<readonly DetBox[]>;
}
