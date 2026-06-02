import { describe, expect, it } from 'vitest';
import { dbPostprocess } from '../../../src/backends/paddle/db_postprocess';

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
