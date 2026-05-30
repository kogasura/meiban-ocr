/**
 * Shared with Python side (`packages/trainer/src/meiban_ocr_trainer/constants.py`).
 * Keep both in sync.
 */

// ===== CTC architecture (旧、CRNN+CTC, npm 0.3.x まで) =====
export const CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' as const;
export const BLANK_IDX = 36;
export const NUM_CLASSES = 37;

// ===== 12-head fixed-length architecture (新、v0.4.0+) =====
// Ericsson serial 専用、出力 13 クラス (`0-9, E, M, ∅`)
export const CHARSET_12H = '0123456789EM' as const;
export const EMPTY_IDX = 12;
export const NUM_CLASSES_12H = 13;
// FIXED_LENGTH: 出力位置数。Ericsson serial は 12 文字だが、ONNX export の
// AdaptiveAvgPool 制約で backbone 出力 W=32 を割り切れる値が必要 → 16 を採用。
// Ericsson 12 文字 + 末尾 4 位置を ∅ で padding。
export const FIXED_LENGTH = 16;

// ===== 共通 =====
export const INPUT_HEIGHT = 32;
export const INPUT_WIDTH = 128;

export const NORM_MEAN = 0.5;
export const NORM_STD = 0.5;
