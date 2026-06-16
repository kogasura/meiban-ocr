/**
 * Backend インターフェース — PaddleOCR (および将来の他実装) を
 * 同一 API で扱うための抽象層。
 *
 * 設計方針:
 * - 各 backend は ImageData → OCRResult[] を返すブラックボックス
 * - 検出器 / preprocess / decode は backend 内に閉じる
 * - URANUS2 側からは `MeibanOCR.create({ backend: 'paddle' })` で選択可能
 * - 既存 API surface (`MeibanOCR.create()`, `recognize()`, `dispose()`) は破壊しない
 *
 * 2026-06-02 dual-backend architecture 導入。
 * 2026-06-16 custom backend (自作 12-head/CRNN) を廃止し paddle 単独に (vendor-setting-client#430)。
 */

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
export type BackendType = 'paddle';

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
  /**
   * rec 推論に渡す bbox の上限 (面積上位 top-K)。 default 8。
   *
   * Why: PP-OCRv4 det は汎用テキスト検出器のため、銘板の複数行＋背景の文字様パターンを
   * 大量に検出し、B(バッチ数)が膨らむと「rec 推論 + box毎JS前処理 + [B×T×6625]CTCデコード」
   * がメインスレッドを長時間占有して UI がフリーズする (custom backend は full-frame で B=1
   * なので無関係)。銘板スキャナは主要テキスト領域だけ読めれば十分なので、面積上位 K 件に絞り
   * メインスレッド負荷を桁で下げる。0 以下なら無制限 (従来挙動)。
   */
  maxRecBoxes?: number;
}

/**
 * MeibanOCR.create() に渡る統合オプション。
 *
 * 現状 backend は paddle のみ。 将来 backend を追加する際は union に戻す。
 */
export type AnyBackendInit = PaddleBackendInit;
