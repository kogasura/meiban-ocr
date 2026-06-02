/**
 * Paddle Backend — PP-OCRv4 mobile EN 版を使った OCR 実装。
 *
 * Phase 2 で完成予定。 現在は stub。
 *
 * 完成時の処理フロー:
 *   image → resize (long side 960px) → PP-OCRv4 det ONNX
 *        → segmentation map → DB postprocess → bbox 配列
 *        → 各 bbox を affine warp で 48×N rectified
 *        → PP-OCRv4 rec EN ONNX → CTC logits
 *        → CTC greedy decode + dict 引き → text
 *        → Ericsson regex + 6 段補正 + 信頼度 gate
 *        → OCRResult[]
 *
 * 依存モデル:
 *   - ch_PP-OCRv4_det.onnx (4.7MB、 言語非依存)
 *   - en_PP-OCRv4_rec.onnx (~5MB、 英数字専用)
 */

import type { Backend, OCRResult, PaddleBackendInit } from './types';

export class PaddleBackend implements Backend {
  static async create(_options: PaddleBackendInit = {}): Promise<PaddleBackend> {
    throw new Error(
      'PaddleBackend is not yet implemented (Phase 2 of dual-backend rollout). ' +
        'Use backend: \'custom\' for now.',
    );
  }

  async recognize(_imageData: ImageData): Promise<OCRResult[]> {
    throw new Error('PaddleBackend.recognize: not implemented');
  }

  async dispose(): Promise<void> {
    // no-op until implementation lands
  }
}
