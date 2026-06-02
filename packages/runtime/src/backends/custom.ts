/**
 * Custom Backend — 自作 12-head fixed-length OCR (兼 旧 CTC CRNN).
 *
 * 内部パイプライン (既存 MeibanOCR.recognize() からそのまま移植):
 *   image → sliding-window 候補生成
 *        → 古典 CV pre-filter (オプション)
 *        → 各 window を 32×128 グレースケール正規化
 *        → ONNX バッチ推論
 *        → output shape による model type 自動判別 (CTC 37 / fixed-head 13)
 *        → デコード + confidence 集約
 *        → 6 段補正パイプライン + regex フィルタ
 *        → NMS (テキスト同一性)
 *
 * 2026-06-02 dual-backend 化に伴い MeibanOCR.ts からこちらに移管。
 */

import * as ort from 'onnxruntime-web';

import { FIXED_LENGTH, NUM_CLASSES, NUM_CLASSES_12H } from '../constants';
import {
  applyCorrectionPipeline,
  ctcGreedyDecodeWithConfidence,
  fixedHeadDecodeWithConfidence,
} from '../decoder';
import { nmsByText, type ScoredDetection } from '../detectors/nms';
import { prefilterBboxes, type PrefilterOptions } from '../detectors/prefilter';
import {
  createSlidingWindowDetector,
  type SlidingWindowOptions,
} from '../detectors/sliding-window';
import type { BBox, DetectorFn } from '../detectors/types';
import { cropAndNormalizeBatch, type RecenterOptions } from '../preprocess';
import { ericsson, VENDOR_PATTERNS, type VendorPattern } from '../vendors';
import { createOrtSession } from './_shared';
import type { Backend, CustomBackendInit, OCRResult } from './types';

const DEFAULT_MIN_CONFIDENCE = 0.5;
const DEFAULT_EPS: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'];

export class CustomBackend implements Backend {
  private readonly session: ort.InferenceSession;
  private readonly detector: DetectorFn;
  private readonly vendor: VendorPattern;
  private readonly minConfidence: number;
  private readonly maxBatchSize: number;
  private readonly prefilterOption: boolean | PrefilterOptions;
  private readonly recenterOption: boolean | RecenterOptions | undefined;

  private constructor(
    session: ort.InferenceSession,
    detector: DetectorFn,
    vendor: VendorPattern,
    options: CustomBackendInit,
  ) {
    this.session = session;
    this.detector = detector;
    this.vendor = vendor;
    this.minConfidence = options.minConfidence ?? DEFAULT_MIN_CONFIDENCE;
    this.maxBatchSize = options.maxBatchSize ?? 64;
    this.prefilterOption = options.prefilter ?? true;
    this.recenterOption = options.recenter;
  }

  static async create(options: CustomBackendInit = {}): Promise<CustomBackend> {
    const vendor = resolveVendor(options.vendor);
    const detector = resolveDetector(options.detector);
    const eps = options.executionProviders ?? DEFAULT_EPS;
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    };
    // バンドル ONNX のデフォルト URL (Vite/Webpack の ?url import で解決)。
    const defaultUrl = (await import('../assets/meiban-ocr-v1.onnx?url')).default;
    const session = await createOrtSession(
      options.modelBytes,
      options.modelUrl,
      defaultUrl,
      sessionOptions,
      'modelUrl',
    );
    return new CustomBackend(session, detector, vendor, options);
  }

  async recognize(imageData: ImageData): Promise<OCRResult[]> {
    const bboxesRaw = await this.detector(imageData);
    if (bboxesRaw.length === 0) return [];

    // 古典 CV pre-filter (default ON)。
    // Phase 2a 実証: pos recall 100% 維持 + 窓数 ~半減 + 推論時間 ~半減。
    let bboxes: BBox[] = bboxesRaw;
    if (this.prefilterOption !== false) {
      const pfOpts: PrefilterOptions =
        this.prefilterOption && typeof this.prefilterOption === 'object'
          ? this.prefilterOption
          : {};
      bboxes = prefilterBboxes(imageData, bboxes, pfOpts);
      if (bboxes.length === 0) return [];
    }

    const scored: ScoredDetection[] = [];

    // recenter: PR #1 で preprocess.ts に追加された window 再センタリング。
    // undefined → cropAndNormalize 側の default (true) に委譲。
    const cropOpts: { recenter?: boolean | RecenterOptions } =
      this.recenterOption === undefined ? {} : { recenter: this.recenterOption };

    for (let i = 0; i < bboxes.length; i += this.maxBatchSize) {
      const batchBoxes = bboxes.slice(i, i + this.maxBatchSize);
      const flat = cropAndNormalizeBatch(imageData, batchBoxes, cropOpts);
      const inputTensor = new ort.Tensor(
        'float32',
        flat,
        [batchBoxes.length, 1, 32, 128],
      );
      const inputName = this.session.inputNames[0]!;
      const outputName = this.session.outputNames[0]!;
      const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
      const out = await this.session.run(feeds);
      const logits = out[outputName]!;
      const [B, T, C] = logits.dims as [number, number, number];

      // Model type auto-detect (旧 v0.3.x CTC ONNX も同じ runtime で動く)。
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
        if (confidence < this.minConfidence) continue;
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

  async dispose(): Promise<void> {
    await this.session.release();
  }
}

function resolveDetector(d?: DetectorFn | SlidingWindowOptions): DetectorFn {
  if (!d) return createSlidingWindowDetector();
  if (typeof d === 'function') return d;
  return createSlidingWindowDetector(d);
}

function resolveVendor(v?: 'ericsson' | VendorPattern): VendorPattern {
  if (!v) return ericsson;
  if (typeof v === 'string') {
    // Why Object.hasOwn: prototype pollution 防御 (MeibanOCR.ts 由来の安全性維持)。
    if (!Object.hasOwn(VENDOR_PATTERNS, v)) {
      throw new Error(`unknown vendor: ${v}`);
    }
    return VENDOR_PATTERNS[v]!;
  }
  return v;
}

function round(x: number, decimals: number): number {
  const k = 10 ** decimals;
  return Math.round(x * k) / k;
}
