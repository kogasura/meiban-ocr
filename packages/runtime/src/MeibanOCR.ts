/**
 * MeibanOCR: 全体画像 → 製造番号抽出のメインクラス。
 *
 * 2026-06-02 dual-backend architecture 化: 内部実装を Backend interface に委譲。
 * 2026-06-16 custom backend を廃止し paddle 単独に (vendor-setting-client#430)。
 * 既存 API surface (`MeibanOCR.create()`, `recognize()`, `dispose()`) は不変。
 *
 * 入力: HTMLCanvas / OffscreenCanvas / ImageBitmap / ImageData
 * 出力: OCRResult[] (Ericsson `E[39]\d{2}MM\d{6}` のみ採用、 6 段補正適用後)
 *
 * Backend:
 *   - `backend: 'paddle'` (default かつ現状唯一): PP-OCRv4 mobile det + rec の 2 段、
 *     訓練不要、 ~15MB
 */

import { createBackend } from './backends/factory';
import type { PaddleBackend } from './backends/paddle';
import type {
  Backend,
  BackendType,
  OCRResult,
  PaddleBackendInit,
  RecognizedLine,
} from './backends/types';
import { imageInputToImageData, type ImageInput } from './preprocess';

/**
 * MeibanOCR.create() に渡すオプション。
 *
 * `backend` で実装を選択 (default かつ現状唯一 'paddle')。
 *
 * Paddle backend で使うフィールド: detModelUrl / recModelUrl / detModelBytes /
 *                                   recModelBytes / dict / 等
 */
export type MeibanOCROptions = {
  /** 認識バックエンド (default 'paddle')。 */
  backend?: BackendType;
} & PaddleBackendInit;

export type { OCRResult, RecognizedLine } from './backends/types';

export class MeibanOCR {
  private readonly backend: Backend;

  private constructor(backend: Backend) {
    this.backend = backend;
  }

  /** Async factory。 backend 選択 + ORT session 初期化を含む。 */
  static async create(options: MeibanOCROptions = {}): Promise<MeibanOCR> {
    const backendType: BackendType = options.backend ?? 'paddle';
    const backend = await createBackend(backendType, options);
    return new MeibanOCR(backend);
  }

  /** 全体画像から製造番号を抽出。 */
  async recognize(image: ImageInput): Promise<OCRResult[]> {
    const imageData = imageInputToImageData(image);
    return this.backend.recognize(imageData);
  }

  /**
   * rec-only 認識。 外部 (古典 CV 等) で切り出し済みの 1 行画像を直接 rec モデルに
   * かけ、 `{ text, confidence }` を返す。
   *
   * `backend: 'paddle'` かつ `recOnly: true` で `create()` した場合のみ利用可能。
   * それ以外の backend / モードで呼ぶと throw する。
   */
  async recognizeLine(image: ImageInput): Promise<RecognizedLine> {
    if (!isRecOnlyPaddleBackend(this.backend)) {
      throw new Error(
        'MeibanOCR.recognizeLine: only available when created with ' +
          "{ backend: 'paddle', recOnly: true }",
      );
    }
    const imageData = imageInputToImageData(image);
    return this.backend.recognizeLine(imageData);
  }

  /** 解放: 内部 backend (ORT session 等) を破棄。 */
  async dispose(): Promise<void> {
    return this.backend.dispose();
  }
}

/**
 * `backend.recognizeLine` の有無で判定する (duck typing)。
 * PaddleBackend は factory.ts の動的 import 経由で生成されるため、
 * `instanceof PaddleBackend` より安全にモジュール非依存で判定できる。
 */
function isRecOnlyPaddleBackend(
  backend: Backend,
): backend is PaddleBackend & {
  recognizeLine(image: ImageData): Promise<RecognizedLine>;
} {
  return (
    typeof (backend as Partial<PaddleBackend>).recognizeLine === 'function'
  );
}
