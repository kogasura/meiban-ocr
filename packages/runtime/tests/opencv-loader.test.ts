import { describe, expect, it } from 'vitest';

// Vitest デフォルトは Node 環境なので window/document は無い。
// loadOpenCv() の SSR ガード (browser 必須エラー) と export 形を確認するに留める。
// 実際の CDN ロードはブラウザ E2E で検証する想定。

describe('loadOpenCv (Node env smoke test)', () => {
  it('is exported from opencv sub-entry', async () => {
    const mod = await import('../src/detectors/opencv-entry');
    expect(typeof mod.loadOpenCv).toBe('function');
    expect(typeof mod.createOpenCvDetector).toBe('function');
  });

  it('throws a clear error when called without window/document (SSR safe)', async () => {
    const { loadOpenCv } = await import('../src/detectors/opencv-entry');
    await expect(loadOpenCv()).rejects.toThrow(/browser environment/);
  });
});
