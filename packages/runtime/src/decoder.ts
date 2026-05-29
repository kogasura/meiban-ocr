/**
 * CTC greedy decoder + confidence aggregator + 6段階補正パイプライン。
 *
 * HANDOFF.md §3 と既存 backend `PlateSerialNumber.php` 互換。
 * - 前処理: NFKC normalize + uppercase, `-` を除去
 * - 6段階補正: 厳格完全一致 → 厳格+O→0 → ... → 厳格部分
 *
 * confidence 集約 (Phase 2c, EasyOCR / production OCR best practice 由来):
 * - **min(geomean(top1), min(top1))** を採用
 * - 旧 arithmetic mean は弱い 1 位置を強い 11 位置で薄める欠陥があり、production で
 *   「pattern OK だが内容違い」を高 conf で accept する原因になっていた
 * - geomean (= 各位置独立の全体正解確率の正規化) と min (= 最弱位置) の **厳しい方**
 *   を採用することで、false-accept 最小化方針 (SECURITY.md) と整合
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

/**
 * Confidence 集約結果。
 *
 * `confidence` が主指標 (= min(geomean, minTop1))。残りは詳細指標で、debug/A-B test 用。
 */
export interface ConfidenceResult {
  /** 主指標。`min(geomean(top1), min(top1))`。0..1。 */
  confidence: number;
  /** top1 prob の geometric mean。EasyOCR 互換指標。 */
  geomean: number;
  /** 各位置の top1 prob のうち最小値。「最弱の 1 位置」。 */
  minTop1: number;
  /** 各位置の (top1 - top2) のうち最小値。「最も迷っている位置の迷い具合」。 */
  minMargin: number;
}

/**
 * 集約: top1 / top2 prob 配列から ConfidenceResult を計算。
 *
 * @param top1Probs  各位置の top1 (argmax) 確率
 * @param top2Probs  各位置の top2 確率 (margin 計算用)
 */
export function aggregateConfidence(
  top1Probs: readonly number[],
  top2Probs: readonly number[],
): ConfidenceResult {
  const n = top1Probs.length;
  if (n === 0) {
    return { confidence: 0, geomean: 0, minTop1: 0, minMargin: 0 };
  }
  if (top2Probs.length !== n) {
    throw new Error(
      `top2Probs length ${top2Probs.length} != top1Probs length ${n}`,
    );
  }
  let minTop1 = Infinity;
  let minMargin = Infinity;
  let logSum = 0;
  for (let i = 0; i < n; i++) {
    const p1 = top1Probs[i]!;
    const p2 = top2Probs[i]!;
    if (p1 < minTop1) minTop1 = p1;
    const margin = p1 - p2;
    if (margin < minMargin) minMargin = margin;
    // log(p) with clamp to avoid -∞ on p=0 (theoretical only; softmax > 0)
    logSum += Math.log(Math.max(p1, 1e-9));
  }
  const geomean = Math.exp(logSum / n);
  const confidence = Math.min(geomean, minTop1);
  return { confidence, geomean, minTop1, minMargin };
}

/**
 * CTC greedy decode + confidence 集約。
 *
 * 非 blank の timestep の softmax から top1/top2 を抽出し、{text, confidence,...} を返す。
 * 「文字内容」と「信頼度」を **同じ 1 pass で**算出する効率重視版。
 *
 * @param logits  (T, C) のフラット配列 (row-major)
 * @param numTimesteps  T
 * @param numClasses  C
 */
export function ctcGreedyDecodeWithConfidence(
  logits: ArrayLike<number>,
  numTimesteps: number,
  numClasses: number,
): { text: string } & ConfidenceResult {
  if (logits.length !== numTimesteps * numClasses) {
    throw new Error(
      `logits length mismatch: expected ${numTimesteps * numClasses}, got ${logits.length}`,
    );
  }
  const out: string[] = [];
  const top1Probs: number[] = [];
  const top2Probs: number[] = [];
  let prev = -1;
  for (let t = 0; t < numTimesteps; t++) {
    const base = t * numClasses;
    // find top 2 logits
    let bestIdx = 0;
    let bestVal = -Infinity;
    let secondVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const v = logits[base + c]!;
      if (v > bestVal) {
        secondVal = bestVal;
        bestVal = v;
        bestIdx = c;
      } else if (v > secondVal) {
        secondVal = v;
      }
    }
    // CTC text reconstruction (collapse repeat + remove blank)
    if (bestIdx !== prev && bestIdx !== BLANK_IDX) {
      out.push(CHARSET[bestIdx]!);
    }
    prev = bestIdx;
    // confidence: skip blank timesteps (文字内容を表す位置のみ)
    if (bestIdx === BLANK_IDX) continue;
    // softmax(top1) と softmax(top2): bestVal を引いて数値安定化
    let sumExp = 0;
    for (let c = 0; c < numClasses; c++) {
      sumExp += Math.exp(logits[base + c]! - bestVal);
    }
    const p1 = 1 / sumExp;
    const p2 = Math.exp(secondVal - bestVal) / sumExp;
    top1Probs.push(p1);
    top2Probs.push(p2);
  }
  const text = out.join('');
  const agg = aggregateConfidence(top1Probs, top2Probs);
  return { text, ...agg };
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
