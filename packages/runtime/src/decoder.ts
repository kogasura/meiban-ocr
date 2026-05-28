/**
 * CTC greedy decoder + 6段階補正パイプライン。
 *
 * HANDOFF.md §3 と既存 backend `PlateSerialNumber.php` 互換。
 * - 前処理: NFKC normalize + uppercase, `-` を除去
 * - 6段階補正: 厳格完全一致 → 厳格+O→0 → ... → 厳格部分
 *
 * Phase 3 で完全実装。Day 1 では greedy decode と前処理だけ実装する。
 */

import { BLANK_IDX, CHARSET } from './constants';
import type { VendorPattern } from './vendors';

/**
 * CTC greedy decode: argmax → blank除去 → 連続重複除去。
 *
 * @param logits  (T, C) のフラット配列 (row-major)
 * @param numTimesteps  T
 * @param numClasses  C
 */
export function ctcGreedyDecode(
  logits: ArrayLike<number>,
  numTimesteps: number,
  numClasses: number,
): string {
  if (logits.length !== numTimesteps * numClasses) {
    throw new Error(
      `logits length mismatch: expected ${numTimesteps * numClasses}, got ${logits.length}`,
    );
  }
  const out: string[] = [];
  let prev = -1;
  for (let t = 0; t < numTimesteps; t++) {
    let bestIdx = 0;
    let bestVal = -Infinity;
    const base = t * numClasses;
    for (let c = 0; c < numClasses; c++) {
      const v = logits[base + c]!;
      if (v > bestVal) {
        bestVal = v;
        bestIdx = c;
      }
    }
    if (bestIdx !== prev && bestIdx !== BLANK_IDX) {
      out.push(CHARSET[bestIdx]!);
    }
    prev = bestIdx;
  }
  return out.join('');
}

/** NFKC + uppercase + `-` 除去。HANDOFF.md §3 の前処理仕様。 */
export function preprocessText(raw: string): string {
  return raw.normalize('NFKC').toUpperCase().replace(/-/g, '');
}

export interface CorrectionResult {
  text: string | null;
  /** 何段階目でマッチしたか。null は未マッチ。 */
  matchStage: number | null;
}

/**
 * 6段階補正パイプライン。Phase 3 で本格実装。
 * Day 1 では 1, 2, 5, 6 の Ericsson 用 stage だけ実装 (stage 3, 4 は将来の他ベンダー向け)。
 */
export function applyCorrectionPipeline(
  raw: string,
  vendor: VendorPattern,
): CorrectionResult {
  const cleaned = preprocessText(raw);

  // Stage 1: 厳格完全一致
  if (vendor.strictRegex.test(cleaned)) {
    return { text: cleaned, matchStage: 1 };
  }
  // Stage 2: 厳格完全一致 + O→0
  const oToZero = cleaned.replace(/O/g, '0');
  if (vendor.strictRegex.test(oToZero)) {
    return { text: oToZero, matchStage: 2 };
  }
  // Stage 3, 4: 寛容パターン (Ericsson 以外、現状未実装)
  // Stage 5: 厳格部分一致 + O→0
  const m5 = oToZero.match(vendor.partialRegex);
  if (m5) {
    return { text: m5[0], matchStage: 5 };
  }
  // Stage 6: 厳格部分一致
  const m6 = cleaned.match(vendor.partialRegex);
  if (m6) {
    return { text: m6[0], matchStage: 6 };
  }
  return { text: null, matchStage: null };
}
