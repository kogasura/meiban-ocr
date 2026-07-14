import { beforeEach, describe, expect, it, vi } from 'vitest';

// onnxruntime-web は WASM 実体を必要とするため、テストでは軽量モックに差し替える。
// InferenceSession.create の呼び出し回数・引数を検証することで、
// recOnly モード時に det モデルの fetch / session 生成がスキップされることを確認する。
const createSessionMock = vi.fn();

vi.mock('onnxruntime-web', () => {
  class TensorMock {
    dims: number[];
    data: Float32Array;
    constructor(_type: string, data: Float32Array, dims: number[]) {
      this.data = data;
      this.dims = dims;
    }
  }
  return {
    InferenceSession: {
      create: createSessionMock,
    },
    Tensor: TensorMock,
  };
});

/** rec 側 ONNX の出力を模したフェイクセッション。CTC decode 可能な logits を返す。 */
function makeFakeRecSession(dict: string[]) {
  const C = dict.length + 2;
  const T = 4;
  return {
    inputNames: ['x'],
    outputNames: ['softmax_0.tmp_0'],
    run: vi.fn(async (feeds: Record<string, { dims: number[] }>) => {
      // 各 timestep で常に blank (idx 0) を argmax にする単純な出力
      // (text は空になるが、confidence/形状だけ検証したいので十分)。
      // 入力 tensor の batch 次元 (dims[0]) に合わせて B 件分の出力を返す
      // (recognizeLines の複数画像バッチ推論を模すため)。
      const B = feeds['x']?.dims[0] ?? 1;
      const data = new Float32Array(B * T * C);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < T; t++) {
          data[b * T * C + t * C] = 0.99; // blank
        }
      }
      return {
        'softmax_0.tmp_0': { data, dims: [B, T, C] },
      };
    }),
    release: vi.fn(async () => {}),
  };
}

function makeFakeDetSession() {
  return {
    inputNames: ['x'],
    outputNames: ['y'],
    run: vi.fn(async () => ({ y: { data: new Float32Array(1), dims: [1, 1, 1, 1] } })),
    release: vi.fn(async () => {}),
  };
}

beforeEach(() => {
  createSessionMock.mockReset();
});

describe('PaddleBackend recOnly mode', () => {
  it('create({ recOnly: true }) skips det session creation (no detModelUrl required)', async () => {
    const dict = ['A', 'B', 'C'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    // det 用の createSession 呼び出しが発生していない (rec の 1 回のみ)
    expect(createSessionMock).toHaveBeenCalledTimes(1);
    expect(backend).toBeInstanceOf(PaddleBackend);
  });

  it('create({ recOnly: true }) does not require detModelUrl/detModelBytes', async () => {
    const dict = ['A'];
    createSessionMock.mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    await expect(
      PaddleBackend.create({
        recOnly: true,
        recModelUrl: 'data:application/octet-stream;base64,AA==',
        dict,
      }),
    ).resolves.toBeDefined();
  });

  it('recognize() throws when created with recOnly: true', async () => {
    const dict = ['A'];
    createSessionMock.mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const fakeImage = { width: 4, height: 4, data: new Uint8ClampedArray(4 * 4 * 4) } as ImageData;
    await expect(backend.recognize(fakeImage)).rejects.toThrow(/recOnly/);
  });

  it('recognizeLine() returns { text, confidence } from a single line image', async () => {
    const dict = ['A', 'B', 'C'];
    createSessionMock.mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const lineImage = {
      width: 100,
      height: 48,
      data: new Uint8ClampedArray(100 * 48 * 4),
    } as ImageData;

    const result = await backend.recognizeLine(lineImage);
    expect(result).toHaveProperty('text');
    expect(result).toHaveProperty('confidence');
    expect(typeof result.text).toBe('string');
    expect(typeof result.confidence).toBe('number');
  });

  it('recognizeLine() returns empty result without invoking the rec session for zero-height input', async () => {
    const dict = ['A', 'B', 'C'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const zeroHeightImage = {
      width: 100,
      height: 0,
      data: new Uint8ClampedArray(0),
    } as ImageData;

    const result = await backend.recognizeLine(zeroHeightImage);
    expect(result).toEqual({ text: '', confidence: 0 });
    expect(recSession.run).not.toHaveBeenCalled();
  });

  it('recognizeLine() returns empty result without invoking the rec session for zero-width input', async () => {
    const dict = ['A', 'B', 'C'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const zeroWidthImage = {
      width: 0,
      height: 48,
      data: new Uint8ClampedArray(0),
    } as ImageData;

    const result = await backend.recognizeLine(zeroWidthImage);
    expect(result).toEqual({ text: '', confidence: 0 });
    expect(recSession.run).not.toHaveBeenCalled();
  });

  it('dispose() releases only the rec session when det session is absent', async () => {
    const dict = ['A'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    await expect(backend.dispose()).resolves.toBeUndefined();
    expect(recSession.release).toHaveBeenCalledTimes(1);
  });
});

describe('PaddleBackend recognizeLines (batch)', () => {
  it('recognizeLines([img]) returns the same result as recognizeLine(img)', async () => {
    const dict = ['A', 'B', 'C'];
    createSessionMock.mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const lineImage = {
      width: 100,
      height: 48,
      data: new Uint8ClampedArray(100 * 48 * 4),
    } as ImageData;

    const single = await backend.recognizeLine(lineImage);
    const batch = await backend.recognizeLines([lineImage]);

    expect(batch).toHaveLength(1);
    expect(batch[0]).toEqual(single);
  });

  it('recognizeLines() returns results in input order for multiple images', async () => {
    const dict = ['A', 'B', 'C'];
    createSessionMock.mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const images = [
      { width: 100, height: 48, data: new Uint8ClampedArray(100 * 48 * 4) } as ImageData,
      { width: 60, height: 48, data: new Uint8ClampedArray(60 * 48 * 4) } as ImageData,
      { width: 200, height: 48, data: new Uint8ClampedArray(200 * 48 * 4) } as ImageData,
    ];

    const results = await backend.recognizeLines(images);
    expect(results).toHaveLength(3);
    for (const r of results) {
      expect(r).toHaveProperty('text');
      expect(r).toHaveProperty('confidence');
    }
  });

  it('recognizeLines([]) returns an empty array without invoking the rec session', async () => {
    const dict = ['A', 'B', 'C'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const results = await backend.recognizeLines([]);
    expect(results).toEqual([]);
    expect(recSession.run).not.toHaveBeenCalled();
  });

  it('recognizeLines() invokes recSession.run exactly once regardless of input count', async () => {
    const dict = ['A', 'B', 'C'];
    const recSession = makeFakeRecSession(dict);
    createSessionMock.mockResolvedValueOnce(recSession);

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    const backend = await PaddleBackend.create({
      recOnly: true,
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    const images = [
      { width: 100, height: 48, data: new Uint8ClampedArray(100 * 48 * 4) } as ImageData,
      { width: 60, height: 48, data: new Uint8ClampedArray(60 * 48 * 4) } as ImageData,
      { width: 200, height: 48, data: new Uint8ClampedArray(200 * 48 * 4) } as ImageData,
      { width: 80, height: 48, data: new Uint8ClampedArray(80 * 48 * 4) } as ImageData,
    ];

    await backend.recognizeLines(images);
    expect(recSession.run).toHaveBeenCalledTimes(1);
  });
});

describe('PaddleBackend default mode (regression)', () => {
  it('create() still requires both det and rec sessions when recOnly is not set', async () => {
    const dict = ['A'];
    createSessionMock
      .mockResolvedValueOnce(makeFakeDetSession())
      .mockResolvedValueOnce(makeFakeRecSession(dict));

    const { PaddleBackend } = await import('../../../src/backends/paddle');
    await PaddleBackend.create({
      detModelUrl: 'data:application/octet-stream;base64,AA==',
      recModelUrl: 'data:application/octet-stream;base64,AA==',
      dict,
    });

    expect(createSessionMock).toHaveBeenCalledTimes(2);
  });

  it('create() throws without detModelUrl/detModelBytes when recOnly is false', async () => {
    const dict = ['A'];
    const { PaddleBackend } = await import('../../../src/backends/paddle');
    await expect(
      PaddleBackend.create({
        recModelUrl: 'data:application/octet-stream;base64,AA==',
        dict,
      }),
    ).rejects.toThrow(/detModelUrl or detModelBytes required/);
  });
});
