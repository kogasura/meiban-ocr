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

describe('ctcGreedyDecodePaddle with charset constraint', () => {
  const dict = ['a', 'b', 'c']; // C = 5 (blank, a, b, c, space)
  const C = 5;

  it('ignores charset when not provided (unchanged behavior)', () => {
    const T = 3;
    const logits = makeLogits([1, 2, 3], T, C);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict);
    expect(result.text).toBe('abc');
  });

  it('forces selection to allowed charset even when a disallowed class has max logit', () => {
    // 各 timestep で 'b' (非許可, idx 2) を最大 logit (0.9) にし、'a' (idx 1) に
    // blank (idx 0) より高い次点確率 (0.5) を与える。charset={'a'} のため
    // 許可されているのは a と blank のみ → b は除外され a が選ばれる。
    // (3 timestep 連続で同一 index 'a' になるため CTC collapse で 1 文字にまとまる)
    const T = 3;
    const C_ = C;
    const logits = new Float32Array(T * C_);
    for (let t = 0; t < T; t++) {
      logits[t * C_ + 0] = 0.001; // blank
      logits[t * C_ + 1] = 0.5; // 'a' (許可、次点)
      logits[t * C_ + 2] = 0.9; // 'b' (非許可だが最大)
    }
    const charset = new Set(['a']);
    const result = ctcGreedyDecodePaddle(logits, T, C_, dict, charset);
    expect(result.text).toBe('a');
  });

  it('always allows blank even under charset constraint', () => {
    // blank (idx 0) を最大 logit にした場合でも charset 制約下で blank は選択可能。
    const T = 3;
    const logits = makeLogits([0, 0, 0], T, C);
    const charset = new Set(['a']);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict, charset);
    expect(result.text).toBe('');
    expect(result.confidence).toBe(0);
  });

  it('ignores charset characters not present in dict (Set ∩ dict)', () => {
    // charset に dict 未収録の 'z' を含めても無視され、'a' のみが有効許可文字になる。
    const T = 2;
    const logits = makeLogits([2, 1], T, C); // 'b', 'a'
    const charset = new Set(['a', 'z']);
    const result = ctcGreedyDecodePaddle(logits, T, C, dict, charset);
    // t=0: 'b' は非許可。 残る候補は blank/a のみでどちらも同じ低確率 → 先に
    //      評価される blank (idx 0) が採用され出力には現れない。
    // t=1: 'a' は許可されそのまま選ばれる。
    expect(result.text).toBe('a');
  });

  it('decodes an Ericsson-like serial correctly under the E/M/digit charset', () => {
    // dict に E, M, 0-9 を含む簡易辞書を用意し、"E305MM503813" を合成する。
    const fullDict = ['E', 'M', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']; // X は非許可のノイズ文字
    const fullC = fullDict.length + 2;
    const charset = new Set(['E', 'M', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    const idxOf = (ch: string) => fullDict.indexOf(ch) + 1;
    const text = 'E305MM503813';
    // 各文字を 1 timestep ずつ生成。 CTC は連続同一 index を collapse するため、
    // "MM" のような連続同一文字の間には blank (idx 0) を 1 timestep 挟む
    // (実際の CRNN 出力でも同一文字が連続する場合は同様に blank で分離される)。
    const seq: number[] = [];
    for (let i = 0; i < text.length; i++) {
      if (i > 0 && text[i] === text[i - 1]) {
        seq.push(0); // blank で分離
      }
      seq.push(idxOf(text[i]!));
    }
    const T = seq.length;
    const logits = makeLogits(seq, T, fullC);
    const result = ctcGreedyDecodePaddle(logits, T, fullC, fullDict, charset);
    expect(result.text).toBe(text);
  });
});
