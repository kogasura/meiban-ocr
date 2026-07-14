/**
 * Paddle Backend — PP-OCRv4 mobile を使った OCR 実装。
 *
 * 処理フロー:
 *   image → resize (long side ≤ 960) → PP-OCRv4 det ONNX
 *        → segmentation map → DB postprocess → bbox 配列
 *        → 各 bbox を rec 入力 (48×320 RGB) に正規化
 *        → PP-OCRv4 rec ONNX → CTC logits
 *        → CTC greedy decode + dict → text (vendor.charset 指定時は許可文字制約)
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
import {
  ctcGreedyDecodeBatch,
  parseDict,
  type CtcDecodeResult,
} from './paddle/ctc_decode';
import { dbPostprocess } from './paddle/db_postprocess';
import {
  preprocessForDet,
  preprocessForRecBatch,
} from './paddle/preprocess';
import type {
  Backend,
  OCRResult,
  PaddleBackendInit,
  RecognizedLine,
} from './types';

const DEFAULT_MIN_CONFIDENCE = 0.5;
const DEFAULT_EPS: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'];

// dict.txt は asset として bundle 同梱、 vite が ?raw で文字列として import 可能。
// (Phase 3 で EN 専用 dict に差し替える場合はここを変更)
import dictText from '../assets/ppocrv4_dict.txt?raw';

export class PaddleBackend implements Backend {
  private readonly detSession: ort.InferenceSession | undefined;
  private readonly recSession: ort.InferenceSession;
  private readonly dict: string[];
  private readonly vendor: VendorPattern;
  private readonly minConfidence: number;
  private readonly detLongSide: number;
  private readonly detBinaryThreshold: number;
  private readonly detMinBoxSize: number;
  private readonly maxRecBoxes: number;
  private readonly recOnly: boolean;

  private constructor(
    detSession: ort.InferenceSession | undefined,
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
    this.detLongSide = options.detLongSide ?? 960;  // E2E実測で 736→960 が検出カバー62→89%/e2e 2倍
    this.detBinaryThreshold = options.detBinaryThreshold ?? 0.3;
    this.detMinBoxSize = options.detMinBoxSize ?? 3;
    this.maxRecBoxes = options.maxRecBoxes ?? 8;
    this.recOnly = options.recOnly ?? false;
  }

  static async create(options: PaddleBackendInit = {}): Promise<PaddleBackend> {
    const vendor = resolveVendor(options.vendor);
    const eps = options.executionProviders ?? DEFAULT_EPS;
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    };

    if (!options.recModelUrl && !options.recModelBytes) {
      throw new Error(
        'PaddleBackend: recModelUrl or recModelBytes required (no default bundle)',
      );
    }

    let detSession: ort.InferenceSession | undefined;
    if (options.recOnly) {
      // rec-only モード: det モデルの fetch / session 生成をスキップする
      // (モバイルのメモリ・初期化時間削減。 外部で検出済みの 1 行画像を渡す用途)
      detSession = undefined;
    } else {
      if (!options.detModelUrl && !options.detModelBytes) {
        throw new Error(
          'PaddleBackend: detModelUrl or detModelBytes required (no default bundle)',
        );
      }
      detSession = await createOrtSession(
        options.detModelBytes,
        options.detModelUrl,
        undefined,
        sessionOptions,
        'detModelUrl',
      );
    }

    const recSession = await createOrtSession(
      options.recModelBytes,
      options.recModelUrl,
      undefined,
      sessionOptions,
      'recModelUrl',
    );

    const dict = options.dict ?? parseDict(dictText);
    return new PaddleBackend(detSession, recSession, dict, vendor, options);
  }

  async recognize(imageData: ImageData): Promise<OCRResult[]> {
    if (this.recOnly || !this.detSession) {
      throw new Error(
        'PaddleBackend.recognize: not available in recOnly mode (det session not loaded). ' +
          'Use recognizeLine() instead.',
      );
    }
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

    // 2.5 box 氾濫対策: 面積上位 top-K に制限してメインスレッド負荷(rec推論+box毎前処理
    //     +[B×T×6625]CTCデコード)を抑える。汎用 det が背景の文字様パターンを大量検出して
    //     B が膨らむと UI がフリーズするため。maxRecBoxes<=0 なら無制限(従来挙動)。
    let recBoxes = bboxes;
    if (this.maxRecBoxes > 0 && bboxes.length > this.maxRecBoxes) {
      recBoxes = [...bboxes]
        .sort((a, b) => (b[2] - b[0]) * (b[3] - b[1]) - (a[2] - a[0]) * (a[3] - a[1]))
        .slice(0, this.maxRecBoxes);
    }

    // 3. Recognition 前処理 + 推論 (batch)
    const recInput = preprocessForRecBatch(imageData, recBoxes);
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

    // 4. CTC greedy decode + dict 引き (vendor に charset があれば制約 decode)
    const decoded = ctcGreedyDecodeBatch(
      recLogits.data as Float32Array,
      B,
      T,
      C,
      this.dict,
      this.vendor.charset,
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
        bbox: recBoxes[i]!,
      });
    }

    // テキスト重複排除 (同じ text を低 conf 側で除去)
    return dedupByText(results);
  }

  /**
   * rec-only 認識。 外部 (古典 CV 等) で切り出し済みの 1 行画像を直接 rec モデルに
   * かける。 `recOnly: true` で生成した場合のみ有効 (det session が無いため)。
   *
   * @param lineImage 切り出し済みの 1 行画像。 高さ 48px 前提 (preprocessForRec が
   *                  アスペクト比を保持して内部リサイズするため、 それ以外の高さでも
   *                  動作はするが、 rec モデルの学習分布に近い 48px 入力を推奨)。
   */
  async recognizeLine(lineImage: ImageData): Promise<RecognizedLine> {
    if (lineImage.width <= 0 || lineImage.height <= 0) {
      return { text: '', confidence: 0 };
    }

    const decoded = await this.runRecBatch([lineImage]);
    return this.decodeToLine(decoded[0]!);
  }

  /**
   * rec-only 認識 (バッチ版)。 外部で切り出し済みの複数行画像を **1 回の
   * `recSession.run`** にまとめて推論する。 `recognizeLine()` を件数分呼ぶ場合と
   * 比べ、 同一フレーム内の複数 ROI をまとめて処理する密集パック用途で律速を解消する。
   *
   * `recOnly: true` で生成した場合のみ有効 (det session が無いため)。
   * 挙動は `recognizeLine()` を入力順に呼んだ場合と一致する
   * (前処理・CTC デコード・補正パイプラインとも同一ロジックを共有)。
   *
   * @param lineImages 切り出し済みの 1 行画像の配列。 空配列なら空配列を返す
   *                    (rec 推論は発生しない)。
   */
  async recognizeLines(lineImages: readonly ImageData[]): Promise<RecognizedLine[]> {
    if (lineImages.length === 0) return [];

    // 0 幅/高さの画像は推論に回さず、結果配列内の該当位置だけ空扱いにする
    // (recognizeLine() の単体挙動と一致させるため)。
    const validIndices: number[] = [];
    const validImages: ImageData[] = [];
    for (let i = 0; i < lineImages.length; i++) {
      const img = lineImages[i]!;
      if (img.width > 0 && img.height > 0) {
        validIndices.push(i);
        validImages.push(img);
      }
    }

    const results: RecognizedLine[] = new Array(lineImages.length).fill(null).map(
      () => ({ text: '', confidence: 0 }),
    );
    if (validImages.length === 0) return results;

    const decoded = await this.runRecBatch(validImages);
    for (let i = 0; i < validIndices.length; i++) {
      results[validIndices[i]!] = this.decodeToLine(decoded[i]!);
    }
    return results;
  }

  /**
   * 複数の 1 行画像を rec 前処理 (batch tensor 化) → `recSession.run` 1 回 →
   * CTC greedy decode まで行う private helper。
   * `recognizeLine` / `recognizeLines` の共通処理 (前処理〜デコード) を集約する。
   */
  private async runRecBatch(
    lineImages: readonly ImageData[],
  ): Promise<CtcDecodeResult[]> {
    // preprocessForRecBatch は「1 枚の画像内の複数 bbox」を想定した API のため、
    // 「複数の独立画像」を渡す本メソッドでは各画像を個別に前処理してから
    // バッチ tensor に連結する。
    const sampleStride = 3 * 48 * 320;
    const batchTensor = new Float32Array(lineImages.length * sampleStride);
    for (let i = 0; i < lineImages.length; i++) {
      const img = lineImages[i]!;
      const bbox: [number, number, number, number] = [0, 0, img.width, img.height];
      const single = preprocessForRecBatch(img, [bbox]);
      batchTensor.set(single.tensor, i * sampleStride);
    }

    const recInputName = this.recSession.inputNames[0]!;
    const recOutputName = this.recSession.outputNames[0]!;
    const recTensor = new ort.Tensor(
      'float32',
      batchTensor,
      [lineImages.length, 3, 48, 320],
    );
    const recOutput = await this.recSession.run({
      [recInputName]: recTensor,
    });
    const recLogits = recOutput[recOutputName]!;
    const [B, T, C] = recLogits.dims as [number, number, number];

    return ctcGreedyDecodeBatch(
      recLogits.data as Float32Array,
      B,
      T,
      C,
      this.dict,
      this.vendor.charset,
    );
  }

  /** CTC decode 結果 1 件に補正パイプラインを適用し RecognizedLine に変換する。 */
  private decodeToLine(decoded: CtcDecodeResult): RecognizedLine {
    const { text: raw, confidence } = decoded;
    const corr = applyCorrectionPipeline(raw, this.vendor);
    // 補正パイプラインが未マッチ (null) の場合は空文字を返す (呼び出し側で
    // text === '' または confidence 閾値未満として弾ける)。
    return { text: corr.text ?? '', confidence: round(confidence, 4) };
  }

  async dispose(): Promise<void> {
    await Promise.all([this.detSession?.release(), this.recSession.release()]);
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
