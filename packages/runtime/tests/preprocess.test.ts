import { describe, expect, it } from 'vitest';
import { cropResizeGrayNormalize, recenterBbox, warpQuadToImage } from '../src/preprocess';

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

describe('warpQuadToImage', () => {
  /** 黒地に回転帯 (白): 中心 (cx,cy)、単位方向 (ux,uy)、長さ len、太さ thick。 */
  function makeRotatedBand(
    width: number,
    height: number,
    cx: number,
    cy: number,
    ux: number,
    uy: number,
    len: number,
    thick: number,
  ): ImageData {
    const img = makeUniformGray(width, height, 0);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const px = x + 0.5 - cx;
        const py = y + 0.5 - cy;
        const u = px * ux + py * uy;
        const n = -px * uy + py * ux;
        if (Math.abs(u) <= len / 2 && Math.abs(n) <= thick / 2) {
          const i = (y * width + x) * 4;
          img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
        }
      }
    }
    return img;
  }

  it('axis-aligned quad: 元画像の部分矩形がそのまま出る', () => {
    const img = makeUniformGray(64, 32, 0);
    // (10,8)-(40,20) を白で塗る
    for (let y = 8; y < 20; y++) {
      for (let x = 10; x < 40; x++) {
        const i = (y * 64 + x) * 4;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
      }
    }
    const out = warpQuadToImage(img, [
      [10, 8],
      [40, 8],
      [40, 20],
      [10, 20],
    ]);
    expect(out.width).toBe(30);
    expect(out.height).toBe(12);
    // 中央は白
    const c = ((6 * 30) + 15) * 4;
    expect(out.data[c]).toBeGreaterThan(250);
  });

  it('回転 quad: 帯に沿った quad を渡すと水平に矯正される', () => {
    // 方向 (0.8, 0.6) ≈ 36.87° の帯、長さ 60、太さ 14
    const img = makeRotatedBand(100, 100, 50, 50, 0.8, 0.6, 60, 14);
    // 帯にぴったり沿う quad (tl, tr, br, bl)
    const ux = 0.8, uy = 0.6;
    const hw = 30, hh = 7;
    const quad: [number, number][] = [
      [50 - ux * hw + uy * hh, 50 - uy * hw - ux * hh],
      [50 + ux * hw + uy * hh, 50 + uy * hw - ux * hh],
      [50 + ux * hw - uy * hh, 50 + uy * hw + ux * hh],
      [50 - ux * hw - uy * hh, 50 - uy * hw + ux * hh],
    ];
    const out = warpQuadToImage(img, quad as never);
    expect(out.width).toBe(60);
    expect(out.height).toBe(14);
    // 矯正後: 中央行はほぼ全幅 白 (帯が水平になっている)
    let rowSum = 0;
    for (let x = 5; x < 55; x++) {
      rowSum += out.data[((7 * 60) + x) * 4]!;
    }
    expect(rowSum / 50).toBeGreaterThan(220);
    // 帯の外 (上下) は黒: warp 後の上端行は band 境界ぼけを除き暗い…
    // band ぴったり quad なので上端行も band 内。代わりに横方向の一様性を確認:
    // 中央行の左端・中央・右端が全部白 = 斜め帯が水平化された証拠
    expect(out.data[((7 * 60) + 5) * 4]).toBeGreaterThan(200);
    expect(out.data[((7 * 60) + 30) * 4]).toBeGreaterThan(200);
    expect(out.data[((7 * 60) + 54) * 4]).toBeGreaterThan(200);
  });

  it('縦長 quad (h/w >= 1.5) は 90° 回転して横長で返る', () => {
    const img = makeUniformGray(64, 64, 128);
    const out = warpQuadToImage(img, [
      [20, 5],
      [30, 5],
      [30, 55],
      [20, 55],
    ]);
    // w=10, h=50 → rot90 で 50×10
    expect(out.width).toBe(50);
    expect(out.height).toBe(10);
  });

  it('画像境界外にはみ出た quad は replicate で埋まり例外を出さない', () => {
    const img = makeUniformGray(32, 32, 200);
    const out = warpQuadToImage(img, [
      [-10, -10],
      [40, -10],
      [40, 20],
      [-10, 20],
    ]);
    expect(out.width).toBe(50);
    expect(out.height).toBe(30);
    expect(out.data[0]).toBe(200); // 範囲外も replicate された一様値
  });
});

describe('cropResizeGrayNormalize (canvas-free crop 経路)', () => {
  it('一様画像は一様な正規化値になる', () => {
    const img = makeUniformGray(300, 80, 255); // 白
    const out = cropResizeGrayNormalize(img.data, 300, 80, [10, 10, 280, 70]);
    expect(out.length).toBe(32 * 128);
    // 白 = (1.0 - 0.5)/0.5 = 1.0
    for (const v of [out[0], out[2000], out[4095]]) {
      expect(v).toBeCloseTo(1.0, 4);
    }
  });

  it('縮小は面積平均 (2x2市松の2倍縮小 = 中間グレー)', () => {
    // 256x64 の 1px 市松模様 → 128x32 に丁度 2x 縮小 → 全画素が (0+255+255+0)/4
    const w = 256, h = 64;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = (x + y) % 2 === 0 ? 0 : 255;
        const i = (y * w + x) * 4;
        data[i] = data[i + 1] = data[i + 2] = v;
        data[i + 3] = 255;
      }
    }
    const out = cropResizeGrayNormalize(data, w, h, [0, 0, w, h]);
    // 平均127.5 → (127.5/255 - 0.5)/0.5 = 0
    expect(out[0]).toBeCloseTo(0, 4);
    expect(out[32 * 128 - 1]).toBeCloseTo(0, 4);
  });

  it('bbox が画像境界をはみ出しても例外を出さず有効領域で処理する', () => {
    const img = makeUniformGray(100, 40, 128);
    const out = cropResizeGrayNormalize(img.data, 100, 40, [-20, -10, 150, 60]);
    expect(out.length).toBe(32 * 128);
    expect(out[100]).toBeCloseTo((128 / 255 - 0.5) / 0.5, 3);
  });

  it('左右で異なる輝度のとき空間配置が保存される', () => {
    // 左半分黒、右半分白
    const w = 400, h = 100;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = x < w / 2 ? 0 : 255;
        const i = (y * w + x) * 4;
        data[i] = data[i + 1] = data[i + 2] = v;
        data[i + 3] = 255;
      }
    }
    const out = cropResizeGrayNormalize(data, w, h, [0, 0, w, h]);
    const row = 16;
    expect(out[row * 128 + 10]).toBeCloseTo(-1.0, 3);  // 左 = 黒
    expect(out[row * 128 + 118]).toBeCloseTo(1.0, 3);  // 右 = 白
  });
});
