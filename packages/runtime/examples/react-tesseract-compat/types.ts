/**
 * OCR Provider abstraction layer.
 *
 * 複数の OCR エンジンを共通インターフェースで切替可能にする。
 * `useOcrProvider(name)` で実装を選択する。
 *
 * 候補:
 * - 'tesseract': 現行 tesseract.js (ブラウザ完結、精度限定)
 * - 'meiban':    @meiban-ocr/runtime (ブラウザ完結、専用CRNN、Ericsson 専門)
 * - 'rapidocr':  (将来) RapidOCR バックエンド呼び出し
 */

export type OcrProviderName = 'tesseract' | 'meiban';

export interface OcrResult {
  /** 認識テキスト (raw or 補正済はプロバイダ依存)。下流の `matchOcr` 互換を保つ。 */
  text: string;
  /** 0-100 スケール (Tesseract に合わせる、しきい値ロジックを変えなくて済む)。 */
  confidence: number;
  /** 任意。MeibanOCR では検出位置を返す、Tesseract では undefined。 */
  bbox?: [number, number, number, number];
}

export interface OcrProvider {
  readonly name: OcrProviderName;
  recognize(image: HTMLCanvasElement): Promise<OcrResult | null>;
  dispose(): Promise<void>;
}

export interface OcrProviderConfig {
  /** vendor 補正に渡す (現状 ericsson のみ)。 */
  vendor?: 'ericsson';
  /** プロバイダ固有のオプション (パススルー)。 */
  [key: string]: unknown;
}
