/**
 * 検出 box の幾何型 (axis-aligned bbox と回転 quad)。
 *
 * paddle backend の det 後処理 (db_postprocess) が出力する box を表現する。
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
 * (NMS / 被覆判定はこちらを使う)。crop は quad の透視変換で行う。
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
