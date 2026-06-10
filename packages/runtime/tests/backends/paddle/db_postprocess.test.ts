import { describe, expect, it } from 'vitest';
import { dbPostprocess, dbPostprocessQuad } from '../../../src/backends/paddle/db_postprocess';

/** segH × segW の seg map を作る。 mask: [(x1,y1,x2,y2), ...] で前景矩形を設定。 */
function makeSegMap(
  segW: number,
  segH: number,
  rects: Array<[number, number, number, number]>,
  fgValue: number = 0.95,
): Float32Array {
  const m = new Float32Array(segW * segH);  // default 0
  for (const [x1, y1, x2, y2] of rects) {
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        m[y * segW + x] = fgValue;
      }
    }
  }
  return m;
}

describe('dbPostprocess', () => {
  it('detects a single rectangular blob and returns its bbox', () => {
    const segW = 32, segH = 16;
    // seg map で (5,5)..(15,10) に前景
    const segMap = makeSegMap(segW, segH, [[5, 5, 15, 10]]);
    // 元画像 segW×4, segH×4 = 128 × 64
    const out = dbPostprocess(segMap, segH, segW, 64, 128, {
      unclipRatio: 1.0,  // expand しない
    });
    expect(out.length).toBe(1);
    const [x1, y1, x2, y2] = out[0]!;
    // 元画像座標 (scaleX = 4, scaleY = 4)
    expect(x1).toBeGreaterThanOrEqual(20); // 5 * 4 = 20
    expect(x2).toBeLessThanOrEqual(60);    // 15 * 4 = 60
    expect(y1).toBeGreaterThanOrEqual(20);
    expect(y2).toBeLessThanOrEqual(40);
  });

  it('returns empty for uniform background', () => {
    const segMap = new Float32Array(32 * 16);  // all zero
    const out = dbPostprocess(segMap, 16, 32, 64, 128);
    expect(out).toEqual([]);
  });

  it('detects multiple separate blobs', () => {
    const segW = 64, segH = 32;
    // 左上に 1 つ、 右下に 1 つ
    const segMap = makeSegMap(segW, segH, [
      [5, 5, 12, 10],
      [40, 20, 50, 28],
    ]);
    const out = dbPostprocess(segMap, segH, segW, segH * 4, segW * 4, {
      unclipRatio: 1.0,
    });
    expect(out.length).toBe(2);
  });

  it('filters small blobs below minBoxSize', () => {
    const segW = 32, segH = 16;
    // 大 blob + 極小 (2x2) blob
    const segMap = makeSegMap(segW, segH, [
      [5, 5, 15, 10],   // 通る
      [20, 12, 22, 14], // minBoxSize=3 で reject
    ]);
    const out = dbPostprocess(segMap, segH, segW, segH * 4, segW * 4, {
      minBoxSize: 3,
      unclipRatio: 1.0,
    });
    expect(out.length).toBe(1);
  });

  it('filters blobs below scoreThreshold', () => {
    const segW = 32, segH = 16;
    // weak signal
    const segMap = makeSegMap(segW, segH, [[5, 5, 15, 10]], 0.4);
    const out = dbPostprocess(segMap, segH, segW, segH * 4, segW * 4, {
      binaryThreshold: 0.3,
      scoreThreshold: 0.6, // 0.4 < 0.6 で reject
    });
    expect(out).toEqual([]);
  });

  it('respects scaleX / scaleY overrides for non-default image sizes', () => {
    const segW = 32, segH = 16;
    const segMap = makeSegMap(segW, segH, [[5, 5, 15, 10]]);
    // 倍率 8x (元画像 256×128)
    const out = dbPostprocess(segMap, segH, segW, 128, 256, {
      unclipRatio: 1.0,
      scaleX: 8,
      scaleY: 8,
    });
    expect(out.length).toBe(1);
    const [x1, , x2] = out[0]!;
    // 5*8=40, 15*8=120 付近
    expect(x1).toBeGreaterThanOrEqual(35);
    expect(x2).toBeLessThanOrEqual(125);
  });

  it('clamps bbox to image bounds (no negative or out-of-image)', () => {
    const segW = 32, segH = 16;
    // 画像端の blob
    const segMap = makeSegMap(segW, segH, [[28, 0, 32, 4]]);
    const out = dbPostprocess(segMap, segH, segW, segH * 4, segW * 4, {
      unclipRatio: 2.0,  // 大幅 expand
    });
    expect(out.length).toBe(1);
    const [x1, y1, x2, y2] = out[0]!;
    expect(x1).toBeGreaterThanOrEqual(0);
    expect(y1).toBeGreaterThanOrEqual(0);
    expect(x2).toBeLessThanOrEqual(segW * 4);
    expect(y2).toBeLessThanOrEqual(segH * 4);
  });
});

/** 回転バー前景: 中心 (cx,cy)、単位方向 (ux,uy)、長さ len、太さ thick。 */
function makeRotatedBarSegMap(
  segW: number,
  segH: number,
  cx: number,
  cy: number,
  ux: number,
  uy: number,
  len: number,
  thick: number,
  fgValue: number = 0.95,
): Float32Array {
  const m = new Float32Array(segW * segH);
  for (let y = 0; y < segH; y++) {
    for (let x = 0; x < segW; x++) {
      const px = x + 0.5 - cx;
      const py = y + 0.5 - cy;
      const u = px * ux + py * uy;
      const n = -px * uy + py * ux;
      if (Math.abs(u) <= len / 2 && Math.abs(n) <= thick / 2) {
        m[y * segW + x] = fgValue;
      }
    }
  }
  return m;
}

describe('dbPostprocessQuad', () => {
  it('axis-aligned blob: quad は rect と同等の外接 bbox を返す', () => {
    const segW = 32, segH = 16;
    const segMap = makeSegMap(segW, segH, [[5, 5, 15, 10]]);
    const out = dbPostprocessQuad(segMap, segH, segW, 64, 128, {
      unclipRatio: 1.0,
    });
    expect(out.length).toBe(1);
    const { bbox, quad } = out[0]!;
    // 前景 x∈[5,15), y∈[5,10)。Vatti unclip は r=1.0 でも d=A/L>0 (本家と同じ):
    // d ≈ 9*4/(2*13) ≈ 1.4 → 元画像座標 (×4) で x∈[14,62], y∈[14,46] 近辺
    expect(bbox[0]).toBeGreaterThanOrEqual(10);
    expect(bbox[2]).toBeLessThanOrEqual(66);
    expect(bbox[1]).toBeGreaterThanOrEqual(10);
    expect(bbox[3]).toBeLessThanOrEqual(48);
    // axis-aligned なので quad の辺も軸並行に近い
    const [tl, tr, , bl] = quad;
    expect(Math.abs(tr[1] - tl[1])).toBeLessThanOrEqual(1);
    expect(Math.abs(bl[0] - tl[0])).toBeLessThanOrEqual(1);
  });

  it('回転 blob: quad の長辺方向が blob の回転角に一致する', () => {
    const segW = 96, segH = 96;
    // 3-4-5 三角形の方向 (0.8, 0.6) = 約36.87°、長さ60、太さ8
    const segMap = makeRotatedBarSegMap(segW, segH, 48, 48, 0.8, 0.6, 60, 8);
    const out = dbPostprocessQuad(segMap, segH, segW, segH, segW, {
      unclipRatio: 1.0,
      scaleX: 1,
      scaleY: 1,
    });
    expect(out.length).toBe(1);
    const [tl, tr, br, bl] = out[0]!.quad;
    const edgeW = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    const edgeH = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
    // 長辺 ≈ 60 + 2d、短辺 ≈ 8 + 2d (d = A*r/L ≈ 3.3、離散化で ±3)
    expect(edgeW).toBeGreaterThan(58);
    expect(edgeW).toBeLessThan(72);
    expect(edgeH).toBeGreaterThan(10);
    expect(edgeH).toBeLessThan(18);
    // 長辺の角度 ≈ atan2(0.6, 0.8) = 36.87° (±5°)
    const angle = (Math.atan2(tr[1] - tl[1], tr[0] - tl[0]) * 180) / Math.PI;
    expect(angle).toBeGreaterThan(31);
    expect(angle).toBeLessThan(43);
    // 一方 axis-aligned 外接 bbox は 60*cos+8*sin ≈ 53px 幅の大きな矩形になる
    const { bbox } = out[0]!;
    expect(bbox[2] - bbox[0]).toBeGreaterThan(48);
  });

  it('unclip: ratio に応じて quad が全辺均等に拡大する (Vatti d = A*r/L)', () => {
    const segW = 64, segH = 32;
    const segMap = makeSegMap(segW, segH, [[10, 10, 40, 20]]); // w=30, h=10
    const base = dbPostprocessQuad(segMap, segH, segW, segH, segW, {
      unclipRatio: 1.0, scaleX: 1, scaleY: 1,
    })[0]!;
    const expanded = dbPostprocessQuad(segMap, segH, segW, segH, segW, {
      unclipRatio: 1.6, scaleX: 1, scaleY: 1,
    })[0]!;
    const w0 = base.bbox[2] - base.bbox[0];
    const w1 = expanded.bbox[2] - expanded.bbox[0];
    const h0 = base.bbox[3] - base.bbox[1];
    const h1 = expanded.bbox[3] - expanded.bbox[1];
    // r=1.0→1.6 の d 増分 ≈ 29*9/(2*38)*0.6 ≈ 2.1 → 2d ≈ 4。幅と高さで増分は同一 (全辺 +d)
    expect(w1 - w0).toBeGreaterThanOrEqual(3);
    expect(w1 - w0).toBeLessThanOrEqual(6);
    expect(Math.abs((w1 - w0) - (h1 - h0))).toBeLessThanOrEqual(2);
  });

  it('score / minBoxSize ゲートは rect 版と同じく機能する', () => {
    const segW = 32, segH = 16;
    const weak = makeSegMap(segW, segH, [[5, 5, 15, 10]], 0.4);
    expect(
      dbPostprocessQuad(weak, segH, segW, 64, 128, { scoreThreshold: 0.6 }),
    ).toEqual([]);
    const tiny = makeSegMap(segW, segH, [[20, 12, 22, 14]]);
    expect(
      dbPostprocessQuad(tiny, segH, segW, 64, 128, { minBoxSize: 3 }),
    ).toEqual([]);
  });

  it('quad の角は画像境界内にクランプされる', () => {
    const segW = 32, segH = 16;
    const segMap = makeSegMap(segW, segH, [[28, 0, 32, 4]]);
    const out = dbPostprocessQuad(segMap, segH, segW, 64, 128, {
      unclipRatio: 2.0,
    });
    expect(out.length).toBe(1);
    for (const [x, y] of out[0]!.quad) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(127);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(63);
    }
  });
});
