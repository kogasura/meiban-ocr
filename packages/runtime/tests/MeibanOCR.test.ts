import { describe, expect, it } from 'vitest';

// Why: MeibanOCR の create/recognize は onnxruntime-web のセッションを必要とし、
// jsdom 環境では WASM ロードに失敗する。ここでは API surface (export, factory名)
// だけを smoke 確認し、推論動作はブラウザ E2E に委ねる。
describe('MeibanOCR public API surface', () => {
  it('exports MeibanOCR class with create() factory', async () => {
    const mod = await import('../src/MeibanOCR');
    expect(typeof mod.MeibanOCR).toBe('function');
    expect(typeof mod.MeibanOCR.create).toBe('function');
  });

  it('OCRResult type is shaped correctly via vendors/decoder', async () => {
    const { ericsson } = await import('../src/vendors');
    const { applyCorrectionPipeline } = await import('../src/decoder');
    const r = applyCorrectionPipeline('E300MM000032', ericsson);
    expect(r.text).toBe('E300MM000032');
  });
});
