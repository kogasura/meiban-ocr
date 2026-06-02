/**
 * Paddle Backend — PP-OCRv4 mobile を使った OCR 実装。
 *
 * 処理フロー:
 *   image → resize (long side ≤ 736) → PP-OCRv4 det ONNX
 *        → segmentation map → DB postprocess → bbox 配列
 *        → 各 bbox を rec 入力 (48×320 RGB) に正規化
 *        → PP-OCRv4 rec ONNX → CTC logits
 *        → CTC greedy decode + dict → text
 *        → Ericsson regex + 6 段補正 + 信頼度 gate
 *        → OCRResult[]
 *
 * モデル:
 *   - ppocrv4_det.onnx (4.7MB、 言語非依存)
 *   - ppocrv4_rec.onnx (~10.3MB、 中国語多言語版、 dict 6623 文字)
 *
 * 注: EN 専用モデル (5MB) は paddle2onnx での変換が必要なため Phase 3 に。
 * 現状は CH 多言語版で動作確認、 Ericsson regex で出力フィルタするので機能的に問題なし。
 */

import * as ort from 'onnxruntime-web';

import { applyCorrectionPipeline } from '../decoder';
import { ericsson, VENDOR_PATTERNS, type VendorPattern } from '../vendors';
import { createOrtSession } from './_shared';
import { ctcGreedyDecodeBatch, parseDict } from './paddle/ctc_decode';
import { dbPostprocess } from './paddle/db_postprocess';
import {
  preprocessForDet,
  preprocessForRecBatch,
} from './paddle/preprocess';
import type { Backend, OCRResult, PaddleBackendInit } from './types';

const DEFAULT_MIN_CONFIDENCE = 0.5;
const DEFAULT_EPS: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'];

// dict.txt は asset として bundle 同梱、 vite が ?raw で文字列として import 可能。
// (Phase 3 で EN 専用 dict に差し替える場合はここを変更)
import dictText from '../assets/ppocrv4_dict.txt?raw';

export class PaddleBackend implements Backend {
  private readonly detSession: ort.InferenceSession;
  private readonly recSession: ort.InferenceSession;
  private readonly dict: string[];
  private readonly vendor: VendorPattern;
  private readonly minConfidence: number;
  private readonly detLongSide: number;
  private readonly detBinaryThreshold: number;
  private readonly detMinBoxSize: number;

  private constructor(
    detSession: ort.InferenceSession,
    recSession: ort.InferenceSession,
    dict: string[],
    vendor: VendorPattern,
    options: PaddleBackendInit,
  ) {
    this.detSession = detSession;
    this.recSession = recSession;
    this.dict = dict;
    this.vendor = vendor;
    this.minConfidence = options.minConfidence ?? DEFAULT_MIN_CONFIDENCE;
    this.detLongSide = options.detLongSide ?? 736;
    this.detBinaryThreshold = options.detBinaryThreshold ?? 0.3;
    this.detMinBoxSize = options.detMinBoxSize ?? 3;
  }

  static async create(options: PaddleBackendInit = {}): Promise<PaddleBackend> {
    const vendor = resolveVendor(options.vendor);
    const eps = options.executionProviders ?? DEFAULT_EPS;
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    };

    if (!options.detModelUrl && !options.detModelBytes) {
      throw new Error(
        'PaddleBackend: detModelUrl or detModelBytes required (no default bundle)',
      );
    }
    if (!options.recModelUrl && !options.recModelBytes) {
      throw new Error(
        'PaddleBackend: recModelUrl or recModelBytes required (no default bundle)',
      );
    }

    const [detSession, recSession] = await Promise.all([
      createOrtSession(
        options.detModelBytes,
        options.detModelUrl,
        undefined,
        sessionOptions,
        'detModelUrl',
      ),
      createOrtSession(
        options.recModelBytes,
        options.recModelUrl,
        undefined,
        sessionOptions,
        'recModelUrl',
      ),
    ]);

    const dict = options.dict ?? parseDict(dictText);
    return new PaddleBackend(detSession, recSession, dict, vendor, options);
  }

  async recognize(imageData: ImageData): Promise<OCRResult[]> {
    // 1. Detection 前処理 + 推論
    const det = preprocessForDet(imageData, this.detLongSide);
    const detInputName = this.detSession.inputNames[0]!;
    const detOutputName = this.detSession.outputNames[0]!;
    const detTensor = new ort.Tensor(
      'float32',
      det.tensor,
      [1, 3, det.height, det.width],
    );
    const detOutput = await this.detSession.run({ [detInputName]: detTensor });
    const segLogits = detOutput[detOutputName]!;
    const [_n, _c, segH, segW] = segLogits.dims as [number, number, number, number];
    const segMap = segLogits.data as Float32Array;

    // 2. DB postprocess → bbox 配列 (元画像座標)
    const bboxes = dbPostprocess(
      segMap,
      segH,
      segW,
      imageData.height,
      imageData.width,
      {
        binaryThreshold: this.detBinaryThreshold,
        minBoxSize: this.detMinBoxSize,
        scaleX: imageData.width / segW,
        scaleY: imageData.height / segH,
      },
    );

    if (bboxes.length === 0) return [];

    // 3. Recognition 前処理 + 推論 (batch)
    const recInput = preprocessForRecBatch(imageData, bboxes);
    const recInputName = this.recSession.inputNames[0]!;
    const recOutputName = this.recSession.outputNames[0]!;
    const recTensor = new ort.Tensor(
      'float32',
      recInput.tensor,
      [recInput.batch, 3, recInput.height, recInput.width],
    );
    const recOutput = await this.recSession.run({
      [recInputName]: recTensor,
    });
    const recLogits = recOutput[recOutputName]!;
    const [B, T, C] = recLogits.dims as [number, number, number];

    // 4. CTC greedy decode + dict 引き
    const decoded = ctcGreedyDecodeBatch(
      recLogits.data as Float32Array,
      B,
      T,
      C,
      this.dict,
    );

    // 5. Ericsson regex + 6 段補正 + 信頼度 gate
    const results: OCRResult[] = [];
    for (let i = 0; i < decoded.length; i++) {
      const { text: raw, confidence } = decoded[i]!;
      if (confidence < this.minConfidence) continue;
      const corr = applyCorrectionPipeline(raw, this.vendor);
      if (!corr.text) continue;
      results.push({
        text: corr.text,
        confidence: round(confidence, 4),
        bbox: bboxes[i]!,
      });
    }

    // テキスト重複排除 (同じ text を低 conf 側で除去)
    return dedupByText(results);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.detSession.release(), this.recSession.release()]);
  }
}

function resolveVendor(v?: 'ericsson' | VendorPattern): VendorPattern {
  if (!v) return ericsson;
  if (typeof v === 'string') {
    if (!Object.hasOwn(VENDOR_PATTERNS, v)) {
      throw new Error(`unknown vendor: ${v}`);
    }
    return VENDOR_PATTERNS[v]!;
  }
  return v;
}

function dedupByText(results: OCRResult[]): OCRResult[] {
  const byText: Map<string, OCRResult> = new Map();
  for (const r of results) {
    const prev = byText.get(r.text);
    if (!prev || r.confidence > prev.confidence) {
      byText.set(r.text, r);
    }
  }
  const out = Array.from(byText.values());
  out.sort((a, b) => b.confidence - a.confidence);
  return out;
}

function round(x: number, decimals: number): number {
  const k = 10 ** decimals;
  return Math.round(x * k) / k;
}
