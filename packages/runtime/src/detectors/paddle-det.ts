/**
 * paddle det (PP-OCRv4 DBNet) を DetectorFn として包むファクトリ。
 *
 * custom backend の detector に渡すと「**paddle det 検出 → custom CRNN 認識**」のハイブリッドになる:
 *   MeibanOCR.create({
 *     backend: 'custom',
 *     modelUrl: '.../meiban-ocr-crnn.onnx',
 *     vendor: 'ericsson',
 *     detector: await createPaddleDetDetector({ detModelUrl: '.../ppocrv4_det.onnx' }),
 *     prefilter: false, recenter: false,   // paddle det の box をそのまま使う
 *     minConfidence: 0.5,
 *   });
 *
 * Why: paddle det は GT シリアル領域を 100% カバー(実測 1515box, IoU0.5/containment 100%)。
 * 一方 paddle **rec** は辞書6623文字で [B,T,6625] CTC デコードが重く(~10s)汎用ゆえ誤発火する。
 * custom rec は 37 クラスで軽く + Ericsson strict regex で非シリアル box を弾くため、
 * 「paddle det + custom rec」は paddle 単体より速く正確。full-frame detector(reticle前提)と違い
 * カメラのフルフレームをそのまま処理できる(複数銘板の同時走査)。
 */

import * as ort from 'onnxruntime-web';

import { createOrtSession } from '../backends/_shared';
import { dbPostprocess } from '../backends/paddle/db_postprocess';
import { preprocessForDet } from '../backends/paddle/preprocess';
import type { BBox, DetectorFn } from './types';

export interface PaddleDetDetectorOptions {
  /** det モデル URL(PP-OCRv4 det)。 detModelBytes 未指定時に必須。 */
  detModelUrl?: string;
  /** det モデルの生バイト(URL より優先)。 */
  detModelBytes?: Uint8Array | ArrayBuffer;
  /** 実行プロバイダ。 default ['webgpu', 'wasm']。 */
  executionProviders?: Array<'webgpu' | 'wasm' | 'webgl'>;
  /** det 入力の長辺リサイズ。 小さいほど速いが recall 低下。 default 736。 */
  detLongSide?: number;
  /** DB 二値化閾値。 default 0.3。 */
  binaryThreshold?: number;
  /** 検出 box 最小サイズ(短辺 px)。 default 3。 */
  minBoxSize?: number;
}

/**
 * paddle det を非同期 DetectorFn に変換する。 内部で det 用 ORT session を1つ生成する。
 */
export async function createPaddleDetDetector(
  options: PaddleDetDetectorOptions,
): Promise<DetectorFn> {
  if (!options.detModelUrl && !options.detModelBytes) {
    throw new Error(
      'createPaddleDetDetector: detModelUrl or detModelBytes required',
    );
  }
  const eps = options.executionProviders ?? ['webgpu', 'wasm'];
  const session = await createOrtSession(
    options.detModelBytes,
    options.detModelUrl,
    undefined,
    { executionProviders: eps, graphOptimizationLevel: 'all' },
    'detModelUrl',
  );
  const detLongSide = options.detLongSide ?? 736;
  const binaryThreshold = options.binaryThreshold ?? 0.3;
  const minBoxSize = options.minBoxSize ?? 3;
  const inputName = session.inputNames[0]!;
  const outputName = session.outputNames[0]!;

  return async (image: ImageData): Promise<BBox[]> => {
    const det = preprocessForDet(image, detLongSide);
    const tensor = new ort.Tensor('float32', det.tensor, [1, 3, det.height, det.width]);
    const out = await session.run({ [inputName]: tensor });
    const seg = out[outputName]!;
    const [, , segH, segW] = seg.dims as [number, number, number, number];
    return dbPostprocess(
      seg.data as Float32Array,
      segH,
      segW,
      image.height,
      image.width,
      {
        binaryThreshold,
        minBoxSize,
        scaleX: image.width / segW,
        scaleY: image.height / segH,
      },
    );
  };
}
