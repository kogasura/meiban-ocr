import { describe, expect, it } from 'vitest';
import {
  ctcGreedyDecodePaddle,
  expectedNumClasses,
  parseDict,
} from '../../../src/backends/paddle/ctc_decode';

describe('parseDict', () => {
  it('splits by newline and strips trailing empty lines', () => {
    const text = 'a\nb\nc\n';
    expect(parseDict(text)).toEqual(['a', 'b', 'c']);
  });

  it('preserves order', () => {
    expect(parseDict('z\ny\nx')).toEqual(['z', 'y', 'x']);
  });

  it('expectedNumClasses adds 2 for blank + space', () => {
    expect(expectedNumClasses(['a', 'b', 'c'])).toBe(5);
  });
});

/** logits builder: 各 timestep で 1 class に 0.9、 残りに均等な低確率を割り振る。 */
function makeLogits(seq: number[], T: number, C: number): Float32Array {
  const lo = 0.001;
  const hi = 0.9;
  const out = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const winner = seq[t]!;
    for (let c = 0; c < C; c++) {
      out[t * C + c] = c === winner ? hi : lo;
    }
  }
  return out;
}

describe('ctcGreedyDecodePaddle', () => {
  const dict = ['a', 'b', 'c'];  // C = 5 (blank, a, b, c, space)
  const C = 5;

  it('decodes simple unique sequence', () => {
    // [a, b, c] → "abc"
    const T = 3;
    const logits = makeLogits([1, 2, 3], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('abc');
    expect(result.confidence).toBeCloseTo(0.9, 5);
  });

  it('collapses repeated classes per CTC rule', () => {
    // [a, a, blank, a] → "aa" (連続 a を collapse、 blank で 区切られて a が復活)
    const T = 4;
    const logits = makeLogits([1, 1, 0, 1], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('aa');
  });

  it('drops blanks', () => {
    // [blank, a, blank, b, blank] → "ab"
    const T = 5;
    const logits = makeLogits([0, 1, 0, 2, 0], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('ab');
  });

  it('maps last index to space', () => {
    // [a, space, b] → "a b" (space は index C-1 = 4)
    const T = 3;
    const logits = makeLogits([1, C - 1, 2], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('a b');
  });

  it('returns empty string and zero confidence for all-blank', () => {
    const T = 3;
    const logits = makeLogits([0, 0, 0], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('');
    expect(result.confidence).toBe(0);
  });

  it('throws on dict size mismatch', () => {
    const T = 1;
    const logits = makeLogits([0], T, C);
    expect(() => ctcGreedyDecodePaddle(logits, T, C, ['a'])).toThrow(/dict mismatch/);
  });
});
