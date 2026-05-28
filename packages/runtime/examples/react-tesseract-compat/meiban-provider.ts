/**
 * MeibanOcrProvider: @meiban-ocr/runtime を OcrProvider 抽象に適合させる。
 *
 * 既存 tesseract.js ベースの useNameplateOcr.ts と同じ呼び出し感を保ちつつ、
 * 切替可能な provider レイヤーに組み込めるようにする。
 */

import { MeibanOCR } from '@meiban-ocr/runtime';
import type { OcrProvider, OcrProviderConfig, OcrResult } from './types';

export async function createMeibanProvider(
  config: OcrProviderConfig = {},
): Promise<OcrProvider> {
  const ocr = await MeibanOCR.create({
    vendor: (config.vendor as 'ericsson') ?? 'ericsson',
  });

  return {
    name: 'meiban',
    async recognize(image: HTMLCanvasElement): Promise<OcrResult | null> {
      const results = await ocr.recognize(image);
      if (results.length === 0) return null;
      // 最 confidence の 1 件を返す (Tesseract API 互換)
      const best = results[0]!;
      return {
        text: best.text,
        // Why x100: uranus2 既存ロジックは Tesseract の 0-100 スケールを前提に
        // しきい値判定する。スケール変換でコールサイトを変えずに済ます。
        confidence: best.confidence * 100,
        bbox: best.bbox,
      };
    },
    async dispose() {
      await ocr.dispose();
    },
  };
}
