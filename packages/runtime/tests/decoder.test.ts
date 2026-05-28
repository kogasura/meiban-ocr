import { describe, expect, it } from 'vitest';
import { BLANK_IDX, CHARSET, NUM_CLASSES } from '../src/constants';
import {
  applyCorrectionPipeline,
  ctcGreedyDecode,
  preprocessText,
} from '../src/decoder';
import { ericsson } from '../src/vendors';

function buildLogits(seq: number[]): Float32Array {
  const T = seq.length;
  const out = new Float32Array(T * NUM_CLASSES);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < NUM_CLASSES; c++) {
      out[t * NUM_CLASSES + c] = -10;
    }
    out[t * NUM_CLASSES + seq[t]!] = 10;
  }
  return out;
}

describe('ctcGreedyDecode', () => {
  it('collapses repeats and removes blanks', () => {
    const A = CHARSET.indexOf('A');
    const B = CHARSET.indexOf('B');
    const C = CHARSET.indexOf('C');
    const seq = [A, A, BLANK_IDX, B, B, BLANK_IDX, BLANK_IDX, C];
    expect(ctcGreedyDecode(buildLogits(seq), seq.length, NUM_CLASSES)).toBe('ABC');
  });

  it('decodes Ericsson serial (with blanks between repeats)', () => {
    // CTC の仕様: 連続同文字は collapse されるので、`M` `M` のような繰り返しは
    // 間に blank を挟まないと 1 つの `M` として出力される。実際のモデルもこの仕様で出力する。
    const target = 'E300MM000032';
    const seq: number[] = [];
    for (let i = 0; i < target.length; i++) {
      const idx = CHARSET.indexOf(target[i]!);
      if (i > 0 && target[i] === target[i - 1]) {
        seq.push(BLANK_IDX);
      }
      seq.push(idx);
    }
    expect(ctcGreedyDecode(buildLogits(seq), seq.length, NUM_CLASSES)).toBe(target);
  });

  it('throws on length mismatch', () => {
    const tiny = new Float32Array(10);
    expect(() => ctcGreedyDecode(tiny, 2, NUM_CLASSES)).toThrow();
  });
});

describe('preprocessText', () => {
  it('NFKC + uppercase + strip dashes', () => {
    expect(preprocessText('e303-mm-500942')).toBe('E300MM000032');
    // 全角数字も NFKC で半角化されることを確認
    expect(preprocessText('Ｅ303ＭＭ500942')).toBe('E300MM000032');
  });
});

describe('applyCorrectionPipeline (Ericsson)', () => {
  it('stage 1: strict exact match', () => {
    expect(applyCorrectionPipeline('E300MM000032', ericsson)).toEqual({
      text: 'E300MM000032',
      matchStage: 1,
    });
  });

  it('stage 1 also handles dashes via preprocess', () => {
    expect(applyCorrectionPipeline('E325-MM-004005', ericsson)).toEqual({
      text: 'E300MM999019',
      matchStage: 1,
    });
  });

  it('stage 2: O→0 fallback', () => {
    // O が混入していて strict にマッチしないケース
    expect(applyCorrectionPipeline('E3O3MM5OO942', ericsson)).toEqual({
      text: 'E300MM000032',
      matchStage: 2,
    });
  });

  it('stage 5: partial match + O→0', () => {
    expect(applyCorrectionPipeline('garbage E3O3MM5OO942 tail', ericsson)).toEqual({
      text: 'E300MM000032',
      matchStage: 5,
    });
  });

  it('stage 5 wins over stage 6 even when O→0 is a no-op (documents pipeline order)', () => {
    // HANDOFF.md §3 のパイプラインは 5 → 6 の順に評価する。O を含まない入力では
    // oToZero === cleaned なので、stage 5 (partial + O→0) が必ず先にヒットする。
    // stage 6 は「stage 5 が null を返したとき」のみ到達可能だが、現行の正規表現では
    // そういう条件は存在しない (= stage 6 は実質 dead-code)。仕様変更時はここで気付ける。
    expect(applyCorrectionPipeline('XXE300MM000032YY', ericsson)).toEqual({
      text: 'E300MM000032',
      matchStage: 5,
    });
  });

  it('returns null when no pattern can be recovered', () => {
    expect(applyCorrectionPipeline('???', ericsson)).toEqual({
      text: null,
      matchStage: null,
    });
  });
});
