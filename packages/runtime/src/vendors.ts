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
}

export const ericsson: VendorPattern = {
  vendorId: 2,
  vendorName: 'ericsson',
  strictRegex: /^E[39]\d{2}MM\d{6}$/,
  partialRegex: /E[39]\d{2}MM\d{6}/,
};

export const VENDOR_PATTERNS: Record<string, VendorPattern> = {
  ericsson,
};
