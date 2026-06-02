/**
 * MeibanOCR: 全体画像 → 製造番号抽出のメインクラス。
 *
 * 2026-06-02 dual-backend architecture 化: 内部実装を Backend interface に委譲。
 * 既存 API surface (`MeibanOCR.create()`, `recognize()`, `dispose()`) は不変。
 *
 * 入力: HTMLCanvas / OffscreenCanvas / ImageBitmap / ImageData
 * 出力: OCRResult[] (Ericsson `E[39]\d{2}MM\d{6}` のみ採用、 6 段補正適用後)
 *
 * Backend 選択:
 *   - `backend: 'custom'` (default): 自作 12-head fixed-length OCR
 *     PR #1 augment v1 + recenter (preprocess.ts:recenterBbox) を内包。
 *   - `backend: 'paddle'`: PP-OCRv4 mobile det + rec の 2 段、 訓練不要、 ~15MB
 */

import { createBackend } from './backends/factory';
import type {
  Backend,
  BackendType,
  CustomBackendInit,
  OCRResult,
  PaddleBackendInit,
} from './backends/types';
import { imageInputToImageData, type ImageInput } from './preprocess';

/**
 * MeibanOCR.create() に渡すオプション。
 *
 * `backend` で実装を選択 (default 'custom')。 backend 別の専用フィールドは
 * 該当 backend のみ参照し、 他フィールドは無視する。
 *
 * Custom backend で使うフィールド: modelUrl / modelBytes / detector / maxBatchSize /
 *                                   prefilter / recenter
 * Paddle backend で使うフィールド: detModelUrl / recModelUrl / detModelBytes /
 *                                   recModelBytes / dict / 等
 */
export type MeibanOCROptions = {
  /** 認識バックエンド (default 'custom')。 */
  backend?: BackendType;
} & (CustomBackendInit | PaddleBackendInit);

export type { OCRResult } from './backends/types';

export class MeibanOCR {
  private readonly backend: Backend;

  private constructor(backend: Backend) {
    this.backend = backend;
  }

  /** Async factory。 backend 選択 + ORT session 初期化を含む。 */
  static async create(options: MeibanOCROptions = {}): Promise<MeibanOCR> {
    const backendType: BackendType = options.backend ?? 'custom';
    const backend = await createBackend(backendType, options);
    return new MeibanOCR(backend);
  }

  /** 全体画像から製造番号を抽出。 */
  async recognize(image: ImageInput): Promise<OCRResult[]> {
    const imageData = imageInputToImageData(image);
    return this.backend.recognize(imageData);
  }

  /** 解放: 内部 backend (ORT session 等) を破棄。 */
  async dispose(): Promise<void> {
    return this.backend.dispose();
  }
}
