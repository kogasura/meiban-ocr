/**
 * MeibanOCR: 全体画像 → 製造番号抽出のメインクラス。
 *
 * 入力: HTMLCanvas / OffscreenCanvas / ImageBitmap / ImageData
 * 出力: OCRResult[] (Ericsson `E[39]\d{2}MM\d{6}` のみ採用)
 *
 * 内部パイプライン:
 *   image → ImageData → sliding-window 候補生成
 *        → 各 window を 32x128 正規化 → CRNN バッチ推論
 *        → CTC greedy decode → 6段補正パイプライン → regex フィルタ → NMS
 */

import * as ort from 'onnxruntime-web';

import { FIXED_LENGTH, NUM_CLASSES, NUM_CLASSES_12H } from './constants';
import {
  applyCorrectionPipeline,
  ctcGreedyDecodeWithConfidence,
  fixedHeadDecodeWithConfidence,
} from './decoder';
import { nmsByText, type ScoredDetection } from './detectors/nms';
import { prefilterBboxes, type PrefilterOptions } from './detectors/prefilter';
import {
  createSlidingWindowDetector,
  type SlidingWindowOptions,
} from './detectors/sliding-window';
import type { BBox, DetectorFn } from './detectors/types';
import { cropAndNormalizeBatch, imageInputToImageData, type ImageInput } from './preprocess';
import { ericsson, VENDOR_PATTERNS, type VendorPattern } from './vendors';

export interface MeibanOCROptions {
  /** vendor 補正パイプライン (default: 'ericsson')。 */
  vendor?: 'ericsson' | VendorPattern;
  /** ORT 実行プロバイダ。優先順、デフォルトは webgpu → wasm。 */
  executionProviders?: Array<'webgpu' | 'wasm' | 'webgl'>;
  /** confidence しきい値。これ未満は除外。default 0.5。 */
  minConfidence?: number;
  /** ONNX モデルの URL (オーバーライド用)。未指定ならバンドル版を使う。 */
  modelUrl?: string;
  /**
   * バンドル版モデルのバイト列。`MeibanOCR.create({ modelBytes })` で明示渡し可。
   * 既定ではバンドル時に生成される `model-bundle.ts` から取り込む。
   */
  modelBytes?: Uint8Array | ArrayBuffer;
  /**
   * 検出器。
   * - 関数 (`DetectorFn`) を渡すと: ImageData → bbox[] を返す責務。OpenCV / 学習済 /
   *   独自実装などすべて差し替え可。`createOpenCvDetector(cv)` 等の helper を使う想定。
   * - オブジェクトを渡すと: 組込 sliding-window のチューニング (旧 API、後方互換)。
   * - 省略時: 組込 sliding-window がデフォルト設定で動く。
   */
  detector?: DetectorFn | SlidingWindowOptions;
  /** 1 バッチ最大件数。デフォルト 64。WebGPU の VRAM 制約対策。 */
  maxBatchSize?: number;
  /**
   * 古典 CV pre-filter (Phase 2a)。エッジ密度 + 局所分散で背景窓を除外、
   * sliding-window の総当たり数を半減して速度改善 + hallucination 削減。
   * - `true` or option: 有効化 (default)
   * - `false`: 無効化 (検出器の bbox をそのまま使う、互換挙動)
   * default 閾値: edgeThreshold=30, varThreshold=100 (Python 実証値)
   */
  prefilter?: boolean | PrefilterOptions;
}

export interface OCRResult {
  text: string;
  confidence: number;
  bbox: [number, number, number, number];
}

const DEFAULT_MIN_CONFIDENCE = 0.5;
const DEFAULT_EPS: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'];

// Why: ORT は `data:` / `blob:` / `https:` / `http:` などを受け付ける。
// 利用側が untrusted な値 (URL query 等) を `modelUrl` に渡したとき、
// `javascript:` / `vbscript:` / `file:` が来ると任意 JS 実行 や local file 読込
// につながる可能性があるため、whitelist で検証する。
const ALLOWED_MODEL_URL_PROTOCOLS = new Set([
  'https:',
  'http:',
  'data:',
  'blob:',
]);

function validateModelUrl(rawUrl: string): void {
  let parsed: URL;
  try {
    const base =
      typeof location !== 'undefined' && location.href
        ? location.href
        : 'http://localhost/';
    parsed = new URL(rawUrl, base);
  } catch {
    throw new Error(`MeibanOCR.create: invalid modelUrl: ${rawUrl}`);
  }
  if (!ALLOWED_MODEL_URL_PROTOCOLS.has(parsed.protocol)) {
    throw new Error(
      `MeibanOCR.create: unsupported protocol "${parsed.protocol}" in modelUrl. ` +
        `Allowed: http, https, data, blob.`,
    );
  }
}

export class MeibanOCR {
  private readonly session: ort.InferenceSession;
  private readonly options: MeibanOCROptions;
  private readonly vendor: VendorPattern;

  /** Async factory。`InferenceSession.create` を内部で await するための形。 */
  static async create(options: MeibanOCROptions = {}): Promise<MeibanOCR> {
    const vendor = MeibanOCR.resolveVendor(options.vendor);
    const eps = options.executionProviders ?? DEFAULT_EPS;
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    };
    let session: ort.InferenceSession;
    if (options.modelBytes) {
      const bytes =
        options.modelBytes instanceof Uint8Array
          ? options.modelBytes
          : new Uint8Array(options.modelBytes);
      session = await ort.InferenceSession.create(bytes, sessionOptions);
    } else if (options.modelUrl) {
      // Security: scheme 検証 (http/https/data/blob のみ許可)
      validateModelUrl(options.modelUrl);
      session = await ort.InferenceSession.create(options.modelUrl, sessionOptions);
    } else {
      // バンドル ONNX。Vite/Webpack の ?url import で URL に解決される。
      // この URL は `dist/assets/meiban-ocr-v1-<hash>.onnx` のような最終パスを指す。
      const modelUrl = (await import('./assets/meiban-ocr-v1.onnx?url')).default;
      session = await ort.InferenceSession.create(modelUrl, sessionOptions);
    }
    return new MeibanOCR(session, options, vendor);
  }

  private readonly detector: DetectorFn;

  private constructor(
    session: ort.InferenceSession,
    options: MeibanOCROptions,
    vendor: VendorPattern,
  ) {
    this.session = session;
    this.options = options;
    this.vendor = vendor;
    this.detector = MeibanOCR.resolveDetector(options.detector);
  }

  /** detector オプションを正規化: 関数ならそのまま、オブジェクトなら組込 sliding-window を構築。 */
  private static resolveDetector(d?: DetectorFn | SlidingWindowOptions): DetectorFn {
    if (!d) return createSlidingWindowDetector();
    if (typeof d === 'function') return d;
    return createSlidingWindowDetector(d);
  }

  private static resolveVendor(v?: 'ericsson' | VendorPattern): VendorPattern {
    if (!v) return ericsson;
    if (typeof v === 'string') {
      // Why Object.hasOwn: `VENDOR_PATTERNS[v]` を直接索引すると
      // v = '__proto__' / 'constructor' / 'toString' 等の prototype member
      // でも値が返ってしまい、後続の `.strictRegex.test(...)` が TypeError
      // になり DoS 経路となる。Object.hasOwn で own-property 限定。
      if (!Object.hasOwn(VENDOR_PATTERNS, v)) {
        throw new Error(`unknown vendor: ${v}`);
      }
      return VENDOR_PATTERNS[v]!;
    }
    return v;
  }

  /** 全体画像から製造番号を抽出。 */
  async recognize(image: ImageInput): Promise<OCRResult[]> {
    const imageData = imageInputToImageData(image);
    const bboxesRaw = await this.detector(imageData);
    if (bboxesRaw.length === 0) return [];

    // 古典 CV pre-filter (default ON)。エッジ密度 + 局所分散で背景窓を除外。
    // Phase 2a 実証: pos recall 100% 維持 + 窓数 ~半減 + 推論時間 ~半減。
    let bboxes: BBox[] = bboxesRaw;
    const pf = this.options.prefilter;
    if (pf !== false) {
      const pfOpts: PrefilterOptions = (pf && typeof pf === 'object') ? pf : {};
      bboxes = prefilterBboxes(imageData, bboxes, pfOpts);
      if (bboxes.length === 0) return [];
    }

    const maxBatch = this.options.maxBatchSize ?? 64;
    const minConf = this.options.minConfidence ?? DEFAULT_MIN_CONFIDENCE;

    const scored: ScoredDetection[] = [];

    for (let i = 0; i < bboxes.length; i += maxBatch) {
      const batchBoxes = bboxes.slice(i, i + maxBatch);
      const flat = cropAndNormalizeBatch(imageData, batchBoxes);
      const inputTensor = new ort.Tensor('float32', flat, [batchBoxes.length, 1, 32, 128]);
      const inputName = this.session.inputNames[0]!;
      const outputName = this.session.outputNames[0]!;
      const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
      const out = await this.session.run(feeds);
      const logits = out[outputName]!;
      const [B, T, C] = logits.dims as [number, number, number];

      // Model type auto-detect:
      //   C=37 (CTC, CHARSET 36 + blank)         → ctcGreedyDecodeWithConfidence
      //   C=13 (fixed-head, CHARSET_12H 12 + ∅)  → fixedHeadDecodeWithConfidence
      // 旧 v0.3.x の ONNX も同じ runtime で動く後方互換性を保つ。
      let decode:
        | typeof ctcGreedyDecodeWithConfidence
        | typeof fixedHeadDecodeWithConfidence;
      if (C === NUM_CLASSES) {
        decode = ctcGreedyDecodeWithConfidence;
      } else if (C === NUM_CLASSES_12H) {
        if (T !== FIXED_LENGTH) {
          throw new Error(
            `fixed-head model expects T=${FIXED_LENGTH}, got T=${T}`,
          );
        }
        decode = fixedHeadDecodeWithConfidence;
      } else {
        throw new Error(
          `unexpected logits C=${C}, expected ${NUM_CLASSES} (CTC) or ${NUM_CLASSES_12H} (fixed-head)`,
        );
      }

      const flatLogits = logits.data as Float32Array;
      for (let b = 0; b < B; b++) {
        const slice = flatLogits.subarray(b * T * C, (b + 1) * T * C);
        const { text: raw, confidence } = decode(slice, T, C);
        const corr = applyCorrectionPipeline(raw, this.vendor);
        if (!corr.text) continue;
        if (confidence < minConf) continue;
        scored.push({
          bbox: batchBoxes[b]! as [number, number, number, number],
          text: corr.text,
          confidence,
        });
      }
    }

    const merged = nmsByText(scored);
    merged.sort((a, b) => b.confidence - a.confidence);
    return merged.map((d) => ({
      text: d.text,
      confidence: round(d.confidence, 4),
      bbox: d.bbox,
    }));
  }

  /** 解放: 内部 ORT session を破棄。 */
  async dispose(): Promise<void> {
    await this.session.release();
  }
}

function round(x: number, decimals: number): number {
  const k = 10 ** decimals;
  return Math.round(x * k) / k;
}
