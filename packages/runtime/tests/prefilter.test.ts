import { describe, expect, it } from 'vitest';
import { prefilterBboxes } from '../src/detectors/prefilter';
import type { BBox } from '../src/detectors/types';

/** ImageData polyfill (Node.js テスト環境用)。 */
function makeImageData(width: number, height: number, data: Uint8ClampedArray): ImageData {
  return { width, height, data, colorSpace: 'srgb' } as ImageData;
}

/** uniform gray でテスト画像を作る。 */
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

/** チェッカーパターン (高エッジ密度 + 高分散)。 */
function makeChecker(width: number, height: number, blockSize: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const v = ((Math.floor(x / blockSize) + Math.floor(y / blockSize)) % 2 === 0) ? 0 : 255;
      data[i + 0] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  return makeImageData(width, height, data);
}

describe('prefilterBboxes', () => {
  it('rejects bboxes on uniform background (low edge + low var)', () => {
    const image = makeUniformGray(256, 128, 128);
    const bboxes: BBox[] = [[10, 10, 138, 42], [50, 50, 178, 82]];
    const out = prefilterBboxes(image, bboxes);
    expect(out).toEqual([]);
  });

  it('passes bboxes on textured area (checkerboard)', () => {
    const image = makeChecker(256, 128, 8);
    const bboxes: BBox[] = [[10, 10, 138, 42]];
    const out = prefilterBboxes(image, bboxes);
    expect(out).toEqual(bboxes);
  });

  it('mixed image: passes textured, rejects uniform', () => {
    // 左半分はテクスチャ、右半分は均一
    const W = 256, H = 128;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        let v: number;
        if (x < W / 2) {
          v = ((Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0) ? 0 : 255;
        } else {
          v = 128;
        }
        data[i + 0] = v;
        data[i + 1] = v;
        data[i + 2] = v;
        data[i + 3] = 255;
      }
    }
    const image = makeImageData(W, H, data);
    const textured: BBox = [10, 10, 110, 42];
    const uniform: BBox = [140, 10, 240, 42];
    const out = prefilterBboxes(image, [textured, uniform]);
    expect(out).toContainEqual(textured);
    expect(out).not.toContainEqual(uniform);
  });

  it('returns empty when input is empty', () => {
    const image = makeChecker(64, 32, 4);
    expect(prefilterBboxes(image, [])).toEqual([]);
  });

  it('respects custom thresholds (very loose passes everything)', () => {
    const image = makeUniformGray(256, 128, 128);
    const bboxes: BBox[] = [[10, 10, 138, 42]];
    const out = prefilterBboxes(image, bboxes, {
      edgeThreshold: 0, varThreshold: 0,
    });
    expect(out).toEqual(bboxes);
  });
});
