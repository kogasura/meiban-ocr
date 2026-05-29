import { describe, expect, it } from 'vitest';
import { BLANK_IDX, CHARSET, NUM_CLASSES } from '../src/constants';
import {
  aggregateConfidence,
  applyCorrectionPipeline,
  ctcGreedyDecode,
  ctcGreedyDecodeWithConfidence,
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
    expect(preprocessText('e300-mm-000032')).toBe('E300MM000032');
    // 全角数字も NFKC で半角化されることを確認
    expect(preprocessText('Ｅ300ＭＭ000032')).toBe('E300MM000032');
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
    // ハイフン除去 + uppercase で strict regex を通過するパターン
    expect(applyCorrectionPipeline('e300-mm-999019', ericsson)).toEqual({
      text: 'E300MM999019',
      matchStage: 1,
    });
  });

  it('stage 2: O→0 fallback', () => {
    // O (英字) が 0 (数字) の位置に混入。O→0 適用で strict にマッチ。
    expect(applyCorrectionPipeline('E3OOMM0OOO32', ericsson)).toEqual({
      text: 'E300MM000032',
      matchStage: 2,
    });
  });

  it('stage 5: partial match + O→0', () => {
    expect(applyCorrectionPipeline('garbage E3OOMM0OOO32 tail', ericsson)).toEqual({
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

// ===== Confidence aggregation (Phase 2c) =====

describe('aggregateConfidence', () => {
  it('returns 0 on empty input', () => {
    const r = aggregateConfidence([], []);
    expect(r.confidence).toBe(0);
    expect(r.geomean).toBe(0);
    expect(r.minTop1).toBe(0);
    expect(r.minMargin).toBe(0);
  });

  it('uniform strong probs give high confidence', () => {
    const r = aggregateConfidence([0.99, 0.99, 0.99], [0.005, 0.005, 0.005]);
    expect(r.confidence).toBeGreaterThan(0.98);
    expect(r.geomean).toBeCloseTo(0.99, 2);
    expect(r.minTop1).toBe(0.99);
    expect(r.minMargin).toBeCloseTo(0.985, 2);
  });

  it('one weak position drags overall confidence down', () => {
    // 弱い 1 位置 (0.30) が「最弱」として支配。
    // 旧 arithmetic mean なら (0.99*11 + 0.30)/12 ≈ 0.93 で通過してしまうケース。
    const top1 = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.30];
    const top2 = top1.map((p) => (1 - p) / 12); // 残りに均等分散の近似
    const r = aggregateConfidence(top1, top2);
    expect(r.minTop1).toBe(0.30);
    expect(r.confidence).toBe(0.30); // = min(geomean, min) = min
  });

  it('all-mid probs give mid confidence', () => {
    const r = aggregateConfidence([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]);
    expect(r.confidence).toBeCloseTo(0.5, 5);
    expect(r.geomean).toBeCloseTo(0.5, 5);
    expect(r.minMargin).toBeCloseTo(0.3, 5);
  });

  it('throws on length mismatch', () => {
    expect(() => aggregateConfidence([0.5, 0.5], [0.5])).toThrow();
  });
});

describe('ctcGreedyDecodeWithConfidence', () => {
  it('returns text + high confidence on strong logits', () => {
    const A = CHARSET.indexOf('A');
    const B = CHARSET.indexOf('B');
    const C = CHARSET.indexOf('C');
    const seq = [A, A, BLANK_IDX, B, B, BLANK_IDX, BLANK_IDX, C];
    const result = ctcGreedyDecodeWithConfidence(buildLogits(seq), seq.length, NUM_CLASSES);
    expect(result.text).toBe('ABC');
    expect(result.confidence).toBeGreaterThan(0.99);
    expect(result.geomean).toBeGreaterThan(0.99);
    expect(result.minTop1).toBeGreaterThan(0.99);
  });

  it('catches one weak position even when other positions are strong', () => {
    const A = CHARSET.indexOf('A');
    const B = CHARSET.indexOf('B');
    const C = CHARSET.indexOf('C');
    // 3 timesteps: A (strong), A (strong, CTC で 1 文字に collapse), B (弱い、C と五分五分)
    const T = 3;
    const logits = new Float32Array(T * NUM_CLASSES);
    for (let i = 0; i < logits.length; i++) logits[i] = -10;
    logits[0 * NUM_CLASSES + A] = 10; // strong
    logits[1 * NUM_CLASSES + A] = 10; // strong (collapses with t=0 A)
    // t=2: B と C をほぼ同点 (B が僅差で勝つ) → top1 prob は ~0.5 になる
    logits[2 * NUM_CLASSES + B] = 1.0;
    logits[2 * NUM_CLASSES + C] = 0.95;
    const result = ctcGreedyDecodeWithConfidence(logits, T, NUM_CLASSES);
    expect(result.text).toBe('AB');
    // t=2 の B の top1 は ~0.51、C の top2 は ~0.49 → minTop1 ~0.51、minMargin ~0.02
    expect(result.minTop1).toBeLessThan(0.6);
    expect(result.minMargin).toBeLessThan(0.1);
    // confidence は min(geomean, minTop1) ≈ minTop1 ≈ 0.51
    expect(result.confidence).toBeLessThan(0.6);
  });

  it('returns 0 confidence when all positions are blank', () => {
    const T = 5;
    const logits = new Float32Array(T * NUM_CLASSES);
    for (let i = 0; i < logits.length; i++) logits[i] = -10;
    for (let t = 0; t < T; t++) logits[t * NUM_CLASSES + BLANK_IDX] = 10;
    const result = ctcGreedyDecodeWithConfidence(logits, T, NUM_CLASSES);
    expect(result.text).toBe('');
    // blank はカウントしない仕様 → 配列空 → conf=0
    expect(result.confidence).toBe(0);
  });

  it('throws on length mismatch', () => {
    const tiny = new Float32Array(10);
    expect(() => ctcGreedyDecodeWithConfidence(tiny, 2, NUM_CLASSES)).toThrow();
  });
});
