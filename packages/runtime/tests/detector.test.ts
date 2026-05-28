import { describe, expect, it } from 'vitest';
import {
  collectWindowBoxes,
  computeDownscale,
  nmsByText,
  type ScoredDetection,
} from '../src/detector';

describe('computeDownscale', () => {
  it('returns 1.0 when smaller than max', () => {
    expect(computeDownscale({ width: 100, height: 200 }, 1024)).toBe(1.0);
  });

  it('scales down by long side', () => {
    expect(computeDownscale({ width: 2048, height: 1024 }, 1024)).toBeCloseTo(0.5);
    expect(computeDownscale({ width: 1024, height: 2048 }, 1024)).toBeCloseTo(0.5);
  });
});

describe('collectWindowBoxes', () => {
  it('produces at least 1 box for a label-shaped input', () => {
    const boxes = collectWindowBoxes({ width: 200, height: 60 });
    expect(boxes.length).toBeGreaterThan(0);
    for (const [x1, y1, x2, y2] of boxes) {
      expect(x2).toBeGreaterThan(x1);
      expect(y2).toBeGreaterThan(y1);
    }
  });

  it('returns empty for input smaller than a single window', () => {
    const boxes = collectWindowBoxes({ width: 50, height: 20 });
    expect(boxes.length).toBe(0);
  });

  it('respects hardLimit', () => {
    expect(() =>
      collectWindowBoxes({ width: 5000, height: 4000 }, {
        strideX: 2,
        strideY: 2,
        hardLimit: 100,
      }),
    ).toThrow(/proposal count/);
  });
});

describe('nmsByText', () => {
  it('removes overlapping detections with same text, keeping highest confidence', () => {
    const dets: ScoredDetection[] = [
      { bbox: [100, 100, 200, 130], text: 'E300MM000032', confidence: 0.95 },
      { bbox: [110, 100, 210, 130], text: 'E300MM000032', confidence: 0.85 },
      { bbox: [300, 100, 400, 130], text: 'E300MM000033', confidence: 0.9 },
    ];
    const kept = nmsByText(dets, 0.3);
    expect(kept).toHaveLength(2);
    expect(kept.map((d) => d.text).sort()).toEqual(['E300MM000032', 'E300MM000033']);
    const survivor = kept.find((d) => d.text === 'E300MM000032')!;
    expect(survivor.confidence).toBeCloseTo(0.95);
  });

  it('keeps non-overlapping detections with same text', () => {
    const dets: ScoredDetection[] = [
      { bbox: [0, 0, 100, 30], text: 'E300MM000032', confidence: 0.9 },
      { bbox: [200, 0, 300, 30], text: 'E300MM000032', confidence: 0.9 },
    ];
    expect(nmsByText(dets)).toHaveLength(2);
  });
});
