/**
 * TesseractOcrProvider: 現行 uranus2 useNameplateOcr.ts のロジックを provider 抽象に
 * リフレームしたもの。`createTesseractProvider({ psm, whitelist })` で再利用可能に。
 *
 * uranus2 側で本ファイルを直接使う想定ではなく、参考実装。
 * 本物の `useNameplateOcr.ts` を provider に書き換えるときは、ここを参照しつつ
 * Next.js の "use client" や React フックは外側 (useOcrProvider) で扱う。
 */

import type { OcrProvider, OcrProviderConfig, OcrResult } from './types';

interface TesseractConfig extends OcrProviderConfig {
  /** PSM 値 (Tesseract.PSM)。デフォルト AUTO=3。 */
  psm?: number;
  /** 文字 whitelist。null で無効化、undefined でデフォルト。 */
  whitelist?: string | null;
  /** traineddata path (default: best モデル CDN)。 */
  langPath?: string;
}

const DEFAULT_WHITELIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-';
const DEFAULT_LANG_PATH = 'https://tessdata.projectnaptha.com/4.0.0_best';

export async function createTesseractProvider(
  config: TesseractConfig = {},
): Promise<OcrProvider> {
  // tesseract.js は dynamic import (重い依存を遅延ロード)
  const { createWorker } = await import('tesseract.js');
  const whitelist =
    config.whitelist === null
      ? ''
      : (config.whitelist ?? DEFAULT_WHITELIST);
  const langPath = config.langPath ?? DEFAULT_LANG_PATH;
  const psm = config.psm ?? 3; // PSM.AUTO

  const worker = await createWorker('eng', 1, { langPath });
  await worker.setParameters({
    tessedit_char_whitelist: whitelist,
    tessedit_pageseg_mode: String(psm),
    user_defined_dpi: '300',
  });

  return {
    name: 'tesseract',
    async recognize(image: HTMLCanvasElement): Promise<OcrResult | null> {
      try {
        const result = await worker.recognize(image);
        return { text: result.data.text, confidence: result.data.confidence };
      } catch {
        return null;
      }
    },
    async dispose() {
      await worker.terminate();
    },
  };
}
