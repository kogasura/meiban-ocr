/**
 * Vendor pattern definitions. HANDOFF.md §2 を参照。
 *
 * Why: 訓練は 36 文字全体で行うが、デコード時にベンダー別パターン制約 +
 * 6段階補正パイプライン (decoder.ts) を適用する。将来のベンダー追加は
 * VENDOR_PATTERNS に entry を足すだけで済む構造に保つ。
 */

export interface VendorPattern {
  vendorId: number;
  vendorName: string;
  /** Strict regex: 全体一致用。 */
  strictRegex: RegExp;
  /** Partial regex: 全文字列の中からシリアルを抜き出す用 (anchor なし)。 */
  partialRegex: RegExp;
  /**
   * CTC デコード時の許可文字集合 (省略可)。
   *
   * Why: strictRegex が示す文字種 (Ericsson は `E[39]\d{2}MM\d{6}` →
   * 実質 {E, M, 0-9} の12種) に、デコード時点で argmax の選択肢を制約すると、
   * 漢字等の巨大多言語辞書への誤読が構造的に消える (悉皆調査で本番30万件が
   * 100%この文字種に一致することを確認済み)。
   *
   * 未指定の vendor / vendor 未指定時は一切制約しない (従来と完全に同一挙動)。
   * dict に存在しない文字は無視される (Set ∩ dict)。 CTC blank は制約対象外
   * (常に選択可能)。
   */
  charset?: ReadonlySet<string>;
}

export const ericsson: VendorPattern = {
  vendorId: 2,
  vendorName: 'ericsson',
  strictRegex: /^E[39]\d{2}MM\d{6}$/,
  partialRegex: /E[39]\d{2}MM\d{6}/,
  charset: new Set(['E', 'M', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
};

export const VENDOR_PATTERNS: Record<string, VendorPattern> = {
  ericsson,
};
