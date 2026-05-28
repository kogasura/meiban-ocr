/**
 * NMS / IoU 共有ユーティリティ。
 */

import type { BBox } from './types';

export interface ScoredDetection {
  bbox: [number, number, number, number];
  text: string;
  confidence: number;
}

export function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const aA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const aB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const u = aA + aB - inter;
  return u > 0 ? inter / u : 0;
}

export function nmsByText(
  detections: ScoredDetection[],
  iouThreshold = 0.3,
): ScoredDetection[] {
  const groups = new Map<string, ScoredDetection[]>();
  for (const d of detections) {
    const arr = groups.get(d.text) ?? [];
    arr.push(d);
    groups.set(d.text, arr);
  }
  const kept: ScoredDetection[] = [];
  for (const arr of groups.values()) {
    arr.sort((a, b) => b.confidence - a.confidence);
    const groupKept: ScoredDetection[] = [];
    for (const d of arr) {
      let suppressed = false;
      for (const k of groupKept) {
        if (iou(d.bbox, k.bbox) >= iouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) groupKept.push(d);
    }
    kept.push(...groupKept);
  }
  return kept;
}
