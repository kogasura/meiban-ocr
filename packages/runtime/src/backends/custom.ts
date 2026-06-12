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

import { FIXED_LENGTH, NUM_CLASSES, NUM_CLASSES_12H, RUNTIME_VERSION } from '../constants';
import {
  applyCorrectionPipeline,
  ctcGreedyDecodeWithConfidence,
  fixedHeadDecodeWithConfidence,
  mergeTailRead,
} from '../decoder';
import { nmsByText, type ScoredDetection } from '../detectors/nms';
import { prefilterBboxes, type PrefilterOptions } from '../detectors/prefilter';
import {
  createSlidingWindowDetector,
  type SlidingWindowOptions,
} from '../detectors/sliding-window';
import { detBoxBBox, detBoxQuad, type DetBox, type DetectorFn } from '../detectors/types';
import {
  cropAndNormalizeBatch,
  cropResizeGrayNormalize,
  warpQuadToImage,
  type RecenterOptions,
} from '../preprocess';
import { ericsson, VENDOR_PATTERNS, type VendorPattern } from '../vendors';
import { createOrtSession } from './_shared';
import type { Backend, CustomBackendInit, OCRResult } from './types';

const DEFAULT_MIN_CONFIDENCE = 0.5;
const DEFAULT_EPS: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'];

export class CustomBackend implements Backend {
  private readonly session: ort.InferenceSession;
  private readonly tailSession: ort.InferenceSession | null;
  private readonly detector: DetectorFn;
  private readonly vendor: VendorPattern;
  private readonly minConfidence: number;
  private readonly tailConfidence: number;
  private readonly maxBatchSize: number;
  private readonly prefilterOption: boolean | PrefilterOptions;
  private readonly recenterOption: boolean | RecenterOptions | undefined;

  private constructor(
    session: ort.InferenceSession,
    tailSession: ort.InferenceSession | null,
    detector: DetectorFn,
    vendor: VendorPattern,
    options: CustomBackendInit,
  ) {
    this.session = session;
    this.tailSession = tailSession;
    this.detector = detector;
    this.vendor = vendor;
    this.minConfidence = options.minConfidence ?? DEFAULT_MIN_CONFIDENCE;
    this.tailConfidence = options.tailConfidence ?? 0.9;
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
    // modelUrl/Bytes 指定時は import しない (base64 chunk ~1.7MB をメモリに載せない)。
    const defaultUrl =
      options.modelBytes || options.modelUrl
        ? undefined
        : (await import('../assets/meiban-ocr-v1.onnx?url')).default;
    const session = await createOrtSession(
      options.modelBytes,
      options.modelUrl,
      defaultUrl,
      sessionOptions,
      'modelUrl',
    );
    // 末尾 2nd-pass モデル (オプション)。未指定なら従来挙動。
    // 'self' = full モデルと同一セッションを再利用 (マルチタスク訓練モデル v11+ 用)。
    // 2本目のセッション (~+109MB) を作らないのでモバイルのメモリに優しい。
    let tailSession: ort.InferenceSession | null = null;
    if (options.tailModelUrl === 'self') {
      tailSession = session;
    } else if (options.tailModelBytes || options.tailModelUrl) {
      tailSession = await createOrtSession(
        options.tailModelBytes,
        options.tailModelUrl,
        undefined,
        sessionOptions,
        'tailModelUrl',
      );
    }
    // ビルド指紋: 実機で「どのビルド・どの構成が動いているか」を console から確定させる
    console.info(
      `[meiban-ocr] custom backend ready (runtime=${RUNTIME_VERSION} eps=${eps.join(',')} ` +
      `tail=${options.tailModelUrl === 'self' ? 'self' : tailSession ? 'separate' : 'off'} ` +
      `maxBatch=${options.maxBatchSize ?? 64})`,
    );
    return new CustomBackend(session, tailSession, detector, vendor, options);
  }

  // recognize の重ね呼び (await されない rAF ループ等) で前処理バッファが滞留しないよう
  // ライブラリ側でも直列化する。アプリ側のガードがあれば実質 no-op。
  private inflight: Promise<OCRResult[]> = Promise.resolve([]);

  async recognize(imageData: ImageData): Promise<OCRResult[]> {
    const next = this.inflight.then(
      () => this.recognizeSerial(imageData),
      () => this.recognizeSerial(imageData),
    );
    this.inflight = next.catch(() => []);
    return next;
  }

  private async recognizeSerial(imageData: ImageData): Promise<OCRResult[]> {
    const boxesRaw = await this.detector(imageData);
    if (boxesRaw.length === 0) return [];

    // 古典 CV pre-filter (default ON)。
    // Phase 2a 実証: pos recall 100% 維持 + 窓数 ~半減 + 推論時間 ~半減。
    // 判定は axis-aligned bbox で行い、quad つき box は quad を保持したまま残す。
    let boxes: readonly DetBox[] = boxesRaw;
    // QuadBox (= 学習済み det 由来) には古典 CV prefilter は不要なので自動スキップ。
    // prefilter はフル解像度の Float32Array を ~12 本確保する (1080p で ~100MB/フレーム)
    // ため、設定し忘れがモバイルのメモリ事故になる。
    const hasQuad = boxes.some((b) => detBoxQuad(b) !== undefined);
    if (!hasQuad && this.prefilterOption !== false) {
      const pfOpts: PrefilterOptions =
        this.prefilterOption && typeof this.prefilterOption === 'object'
          ? this.prefilterOption
          : {};
      const kept = new Set(
        prefilterBboxes(imageData, boxes.map(detBoxBBox), pfOpts),
      );
      boxes = boxes.filter((b) => kept.has(detBoxBBox(b)));
      if (boxes.length === 0) return [];
    }

    // recenter: PR #1 で preprocess.ts に追加された window 再センタリング。
    // undefined → cropAndNormalize 側の default (true) に委譲。
    const cropOpts: { recenter?: boolean | RecenterOptions } =
      this.recenterOption === undefined ? {} : { recenter: this.recenterOption };

    // Phase 1: full read (従来どおり)。補正・採点は tail 2nd-pass 後に行うため
    // raw decode を保持する。
    interface RawRead {
      box: DetBox;
      raw: string;
      confidence: number;
      charTimesteps: number[] | null;
      numTimesteps: number;
    }
    const raws: RawRead[] = [];
    for (let i = 0; i < boxes.length; i += this.maxBatchSize) {
      const batchBoxes = boxes.slice(i, i + this.maxBatchSize);
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
      const isCtc = C === NUM_CLASSES;
      if (!isCtc && C !== NUM_CLASSES_12H) {
        throw new Error(
          `unexpected logits C=${C}, expected ${NUM_CLASSES} (CTC) or ${NUM_CLASSES_12H} (fixed-head)`,
        );
      }
      if (!isCtc && T !== FIXED_LENGTH) {
        throw new Error(`fixed-head model expects T=${FIXED_LENGTH}, got T=${T}`);
      }

      const flatLogits = logits.data as Float32Array;
      for (let b = 0; b < B; b++) {
        const slice = flatLogits.subarray(b * T * C, (b + 1) * T * C);
        if (isCtc) {
          const { text, confidence, charTimesteps } =
            ctcGreedyDecodeWithConfidence(slice, T, C);
          raws.push({
            box: batchBoxes[b]!, raw: text, confidence,
            charTimesteps, numTimesteps: T,
          });
        } else {
          const { text, confidence } = fixedHeadDecodeWithConfidence(slice, T, C);
          raws.push({
            box: batchBoxes[b]!, raw: text, confidence,
            charTimesteps: null, numTimesteps: T,
          });
        }
      }
    }

    // Phase 2: 末尾 2nd-pass (tail モデルが設定されている場合のみ)。
    if (this.tailSession) {
      await this.applyTailPass(imageData, raws);
    }

    // Phase 3: 補正パイプライン + confidence gate + NMS
    const scored: ScoredDetection[] = [];
    for (const r of raws) {
      const corr = applyCorrectionPipeline(r.raw, this.vendor);
      if (!corr.text) continue;
      if (r.confidence < this.minConfidence) continue;
      scored.push({
        bbox: detBoxBBox(r.box) as [number, number, number, number],
        text: corr.text,
        confidence: r.confidence,
      });
    }

    const merged = nmsByText(scored);
    merged.sort((a, b) => b.confidence - a.confidence);
    return merged.map((d) => ({
      text: d.text,
      confidence: round(d.confidence, 4),
      bbox: d.bbox,
    }));
  }

  /**
   * 末尾 2nd-pass: 12文字 read (CTC) かつ quad つき box に対し、CTC アライメントで
   * 末尾4文字領域を再 crop → tail モデルで再読 → mergeTailRead で末尾2文字を差し替え。
   *
   * Why: 誤読の95%が pos10/11 に集中 (full crop で末尾1文字≈10px に潰れるため)。
   * 右端再 crop は1文字≈32px。held-out 実測 E2E +0.92pt / 偽発火微減 / 新規発火なし。
   */
  private async applyTailPass(
    imageData: ImageData,
    raws: Array<{
      box: DetBox; raw: string; confidence: number;
      charTimesteps: number[] | null; numTimesteps: number;
    }>,
  ): Promise<void> {
    interface TailJob { idx: number; input: Float32Array }
    const jobs: TailJob[] = [];
    for (let i = 0; i < raws.length; i++) {
      const r = raws[i]!;
      if (r.raw.length !== 12 || !r.charTimesteps || r.charTimesteps.length !== 12) {
        continue;
      }
      const quad = detBoxQuad(r.box);
      if (!quad) continue;
      // full read の crop 領域を再現 (透視変換)。tail だけ再warpするコストは
      // 12文字 read を出した box に限られるため軽微。
      const warped = warpQuadToImage(imageData, quad);
      // 末尾4文字の開始位置 = pos7/pos8 の emission 中点 (Python 実装と同一)
      const t8 = r.charTimesteps[8]!;
      const t7 = r.charTimesteps[7]!;
      const x0 = Math.round(((t8 + t7) / 2 / r.numTimesteps) * warped.width);
      if (warped.width - x0 < 8) continue;
      jobs.push({
        idx: i,
        input: cropResizeGrayNormalize(
          warped.data, warped.width, warped.height,
          [x0, 0, warped.width, warped.height],
        ),
      });
    }
    if (jobs.length === 0) return;

    const stride = 32 * 128;
    const inputName = this.tailSession!.inputNames[0]!;
    const outputName = this.tailSession!.outputNames[0]!;
    for (let i = 0; i < jobs.length; i += this.maxBatchSize) {
      const batch = jobs.slice(i, i + this.maxBatchSize);
      const flat = new Float32Array(batch.length * stride);
      batch.forEach((j, n) => flat.set(j.input, n * stride));
      const out = await this.tailSession!.run({
        [inputName]: new ort.Tensor('float32', flat, [batch.length, 1, 32, 128]),
      });
      const logits = out[outputName]!;
      const [B, T, C] = logits.dims as [number, number, number];
      const flatLogits = logits.data as Float32Array;
      for (let b = 0; b < B; b++) {
        const slice = flatLogits.subarray(b * T * C, (b + 1) * T * C);
        const { text, confidence } = ctcGreedyDecodeWithConfidence(slice, T, C);
        const r = raws[batch[b]!.idx]!;
        r.raw = mergeTailRead(r.raw, text, confidence, this.tailConfidence);
      }
    }
  }

  private disposed = false;

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.session.release();
    // 'self' 共有時は同一セッションなので二重 release しない
    if (this.tailSession && this.tailSession !== this.session) {
      await this.tailSession.release();
    }
    // detector が dispose を持つ場合 (createPaddleDetDetector) は連鎖解放。
    // det セッションは実測 +170〜270MB あり、解放経路が無いと再マウント毎にリークする。
    const d = this.detector as { dispose?: () => Promise<void> };
    if (typeof d.dispose === 'function') {
      await d.dispose();
    }
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
