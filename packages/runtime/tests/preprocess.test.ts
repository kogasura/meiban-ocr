import { describe, expect, it } from 'vitest';
import { recenterBbox } from '../src/preprocess';

/** ImageData polyfill (Node.js テスト環境用、prefilter.test.ts と同じパターン)。 */
function makeImageData(width: number, height: number, data: Uint8ClampedArray): ImageData {
  return { width, height, data, colorSpace: 'srgb' } as ImageData;
}

/** uniform gray (背景) で初期化。 */
function makeUniformGray(width: number, height: number, value: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4 + 0] = value;
    data[i * 4 + 1] = value;
    data[i * 4 + 2] = value;
    data[i * 4 + 3] = 255;
  }
  return makeImageData(width, height, data);
}

/**
 * 任意の x 範囲に text 風の縦線パターン (高水平勾配) を描画。
 * 2px 黒 → 2px 白 の交互ブロック (1D 水平差分 |gray[x+1]-gray[x-1]| が
 * 内部列で常に大きな値を出すように設計、 1px 交互だと x±1 が同色で勾配=0 になる)。
 */
function drawTextBand(
  image: ImageData,
  textX1: number,
  textX2: number,
  textY1: number,
  textY2: number,
): void {
  for (let y = textY1; y < textY2; y++) {
    for (let x = textX1; x < textX2; x++) {
      const i = (y * image.width + x) * 4;
      const v = (((x - textX1) >> 1) & 1) === 0 ? 0 : 255;
      image.data[i + 0] = v;
      image.data[i + 1] = v;
      image.data[i + 2] = v;
      image.data[i + 3] = 255;
    }
  }
}

describe('recenterBbox', () => {
  it('text 中央: シフトせず no-op に近い', () => {
    // 256x32 画像、 text は [108, 148) の中央 40px
    const img = makeUniformGray(256, 32, 128);
    drawTextBand(img, 108, 148, 8, 24);
    // bbox は [64, 0, 192, 32] (幅 128、text を中央に含む)
    const out = recenterBbox(img, [64, 0, 192, 32]);
    // 元 bbox 中心は 128、text 中心は 128 → ほぼシフトなし
    expect(Math.abs(out[0] - 64)).toBeLessThanOrEqual(2);
    expect(out[1]).toBe(0);
    expect(out[2] - out[0]).toBe(128);
    expect(out[3]).toBe(32);
  });

  it('text が窓の右端に寄っている: bbox が右にシフト', () => {
    // 256x32 画像、 text は [120, 160) (= bbox [40,168) 中央 104 より右)
    const img = makeUniformGray(256, 32, 128);
    drawTextBand(img, 120, 160, 8, 24);
    // bbox [40, 0, 168, 32]、 幅 128、 中心 104
    const out = recenterBbox(img, [40, 0, 168, 32]);
    // text 中心 ~140 に bbox 中心を寄せたい → newX1 ≈ 76
    const newCenter = (out[0] + out[2]) / 2;
    expect(newCenter).toBeGreaterThan(125);
    expect(newCenter).toBeLessThan(155);
    // 幅は変えない
    expect(out[2] - out[0]).toBe(128);
  });

  it('text が窓の左端に寄っている: bbox が左にシフト', () => {
    // 256x32 画像、 text は [60, 100) (= bbox [80,208) 中央 144 より左)
    const img = makeUniformGray(256, 32, 128);
    drawTextBand(img, 60, 100, 8, 24);
    const out = recenterBbox(img, [80, 0, 208, 32]);
    // text 中心 ~80 へ bbox 中心を寄せる、ただし maxShiftRatio=0.4 で ±51 px に制限
    // 旧 bbox 中心 144 → 期待値 max(80, 144 - 51) = 93 付近
    const newCenter = (out[0] + out[2]) / 2;
    expect(newCenter).toBeLessThan(144); // 左にシフト
    expect(newCenter).toBeGreaterThan(70); // ただし行き過ぎない
    expect(out[2] - out[0]).toBe(128);
  });

  it('uniform 背景: activity 不足で no-op', () => {
    const img = makeUniformGray(256, 32, 128);
    const orig: [number, number, number, number] = [64, 0, 192, 32];
    const out = recenterBbox(img, orig);
    expect(out).toEqual(orig);
  });

  it('bbox が画像右端寄り: シフト後も画像内にクランプ', () => {
    // 画像幅 200、 bbox [80, 0, 200, 32]、 text は bbox 右半に
    const img = makeUniformGray(200, 32, 128);
    drawTextBand(img, 160, 195, 8, 24);
    const out = recenterBbox(img, [80, 0, 200, 32]);
    // newX1 + bw <= W = 200
    expect(out[0]).toBeGreaterThanOrEqual(0);
    expect(out[2]).toBeLessThanOrEqual(200);
    expect(out[2] - out[0]).toBe(120);
  });

  it('shift 量は maxShiftRatio で制限される', () => {
    // text を search 範囲内に置きつつ、bbox 中心からは大きく離す
    // bbox [200, 0, 328, 32]、 expand 0.3 → search [162, 366)、 search 内なら検出可能
    // text [162, 195) を置く (search 左端)、 text 中心 ~178、 bbox 中心 264 → dx = -86
    // maxShiftRatio=0.4 で ±51 に制限 → newX1 = 200 - 51 = 149 付近
    const img = makeUniformGray(512, 32, 128);
    drawTextBand(img, 162, 195, 8, 24);
    const out = recenterBbox(img, [200, 0, 328, 32]);
    expect(out[0]).toBeGreaterThanOrEqual(140);
    expect(out[0]).toBeLessThanOrEqual(160);
  });

  it('totalActivity が 0 なら minActivity=0 でも no-op (uniform 領域)', () => {
    const img = makeUniformGray(256, 32, 128);
    // minActivity=0 でも totalActivity <= 0 の fallback が effective に保つ。
    const orig: [number, number, number, number] = [64, 0, 192, 32];
    const out = recenterBbox(img, orig, { minActivity: 0 });
    expect(out).toEqual(orig);
  });

  it('退化ケース: bbox が極小だと no-op', () => {
    const img = makeUniformGray(64, 32, 128);
    const orig: [number, number, number, number] = [10, 5, 11, 6];
    const out = recenterBbox(img, orig);
    expect(out).toEqual(orig);
  });
});
