/**
 * `useOcrProvider(name, config)` — OCR エンジンを切替可能にする React フック。
 *
 * 既存の `useNameplateOcr` (Tesseract 専用) を完全に置換できる API を保ちつつ、
 * `name='tesseract' | 'meiban'` で実装を切替。`name` を React state にすると
 * UI トグルから差し替え可能。
 *
 * 設計:
 * - active flag で unmount/切替時に走る古い recognize を遮断
 * - dispose は前 provider に対して必ず呼ぶ
 * - initError は文字列で expose (UI に出しやすい)
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { createMeibanProvider } from './meiban-provider';
import { createTesseractProvider } from './tesseract-provider';
import type { OcrProvider, OcrProviderConfig, OcrProviderName, OcrResult } from './types';

export interface UseOcrProviderReturn {
  recognize: (image: HTMLCanvasElement) => Promise<OcrResult | null>;
  isReady: boolean;
  initError: string | null;
  /** 現在アクティブな provider 名 (debug 表示などに)。 */
  activeName: OcrProviderName | null;
}

export function useOcrProvider(
  name: OcrProviderName,
  config: OcrProviderConfig = {},
): UseOcrProviderReturn {
  const providerRef = useRef<OcrProvider | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<OcrProviderName | null>(null);

  // config を依存に入れると毎レンダーで再生成されるので、JSON 化してキー化
  const configKey = JSON.stringify(config);

  useEffect(() => {
    let active = true;
    setIsReady(false);
    setInitError(null);

    (async () => {
      try {
        const provider =
          name === 'meiban'
            ? await createMeibanProvider(config)
            : await createTesseractProvider(config);
        if (!active) {
          await provider.dispose();
          return;
        }
        providerRef.current = provider;
        setActiveName(provider.name);
        setIsReady(true);
      } catch (e) {
        setInitError(e instanceof Error ? e.message : 'OCR初期化に失敗しました');
      }
    })();

    return () => {
      active = false;
      const p = providerRef.current;
      providerRef.current = null;
      setActiveName(null);
      if (p) void p.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, configKey]);

  const recognize = useCallback(
    async (image: HTMLCanvasElement): Promise<OcrResult | null> => {
      const p = providerRef.current;
      if (!p) return null;
      return p.recognize(image);
    },
    [],
  );

  return { recognize, isReady, initError, activeName };
}
