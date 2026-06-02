/**
 * Backend インターフェース — Custom 12-head と PaddleOCR (および将来の他実装) を
 * 同一 API で扱うための抽象層。
 *
 * 設計方針:
 * - 各 backend は ImageData → OCRResult[] を返すブラックボックス
 * - 検出器 / preprocess / decode は backend 内に閉じる
 * - URANUS2 側からは `MeibanOCR.create({ backend: 'custom' | 'paddle' })` で選択可能
 * - 既存 API surface (`MeibanOCR.create()`, `recognize()`, `dispose()`) は破壊しない
 *
 * 2026-06-02 dual-backend architecture 導入。
 */

import type { PrefilterOptions } from '../detectors/prefilter';
import type { SlidingWindowOptions } from '../detectors/sliding-window';
import type { DetectorFn } from '../detectors/types';
import type { VendorPattern } from '../vendors';

/** Backend が返す共通 result 型 (MeibanOCR.ts と一致)。 */
export interface OCRResult {
  text: string;
  confidence: number;
  bbox: [number, number, number, number];
}

/** すべての backend が満たすインターフェース。 */
export interface Backend {
  /** 1 frame → 検出された各 plate の text + conf + bbox。 */
  recognize(image: ImageData): Promise<OCRResult[]>;

  /** 内部 ORT session 等の解放。 */
  dispose(): Promise<void>;
}

/** どの backend を使うかの discriminator。 */
export type BackendType = 'custom' | 'paddle';

/** 全 backend 共通のオプション。 */
export interface CommonBackendOptions {
  /** ORT 実行プロバイダ。優先順、デフォルトは webgpu → wasm。 */
  executionProviders?: Array<'webgpu' | 'wasm' | 'webgl'>;
  /** confidence しきい値。これ未満は除外。default 0.5。 */
  minConfidence?: number;
  /** vendor 補正パイプライン (default: 'ericsson')。 */
  vendor?: 'ericsson' | VendorPattern;
}

/**
 * Custom backend (既存 12-head fixed-length OCR) の初期化オプション。
 * 既存 MeibanOCROptions の中身そのまま。
 */
export interface CustomBackendInit extends CommonBackendOptions {
  /** ONNX モデルの URL (オーバーライド用)。未指定ならバンドル版を使う。 */
  modelUrl?: string;
  /** バンドル版モデルのバイト列。 */
  modelBytes?: Uint8Array | ArrayBuffer;
  /**
   * 検出器。
   * - 関数 (`DetectorFn`) を渡すと: ImageData → bbox[] を返す責務
   * - オブジェクトを渡すと: 組込 sliding-window のチューニング
   * - 省略時: 組込 sliding-window がデフォルト設定で動く
   */
  detector?: DetectorFn | SlidingWindowOptions;
  /** 1 バッチ最大件数。デフォルト 64。WebGPU の VRAM 制約対策。 */
  maxBatchSize?: number;
  /**
   * 古典 CV pre-filter (Phase 2a)。エッジ密度 + 局所分散で背景窓を除外。
   * - `true` or option: 有効化 (default)
   * - `false`: 無効化 (検出器の bbox をそのまま使う)
   */
  prefilter?: boolean | PrefilterOptions;
}

/**
 * Paddle backend (PP-OCRv4 mobile EN 版) の初期化オプション。
 * det モデル + rec モデルの 2 つを要する。
 */
export interface PaddleBackendInit extends CommonBackendOptions {
  /** Detection モデル ONNX の URL (PP-OCRv4 det)。 */
  detModelUrl?: string;
  /** Recognition モデル ONNX の URL (PP-OCRv4 rec EN)。 */
  recModelUrl?: string;
  /** Detection モデルバイト列 (URL の代替)。 */
  detModelBytes?: Uint8Array | ArrayBuffer;
  /** Recognition モデルバイト列 (URL の代替)。 */
  recModelBytes?: Uint8Array | ArrayBuffer;
  /**
   * Recognition の文字辞書 override。
   * 未指定なら同梱の英数字辞書 (95 文字、 `assets/ppocrv4_dict_en.txt`) を使う。
   */
  dict?: string[];
  /**
   * Detection の長辺リサイズサイズ。 default 960。
   * 小さくすると検出精度が落ちるが速度が上がる。
   */
  detLongSide?: number;
  /**
   * Detection の二値化閾値。 default 0.3。 0-1 の範囲。
   */
  detBinaryThreshold?: number;
  /**
   * 検出 bbox の最低サイズ (短辺 px)。 default 3。
   */
  detMinBoxSize?: number;
}

/**
 * MeibanOCR.create() に渡る統合オプション。
 *
 * `backend` でどの実装を使うか選ぶ。 default 'custom' (既存挙動 100% 互換)。
 * backend 別の専用フィールドは該当 backend のみ参照する (他は無視)。
 */
export type AnyBackendInit = CustomBackendInit | PaddleBackendInit;
