/**
 * Shared with Python side (`packages/trainer/src/meiban_ocr_trainer/constants.py`).
 * Keep both in sync.
 */

export const CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' as const;
export const BLANK_IDX = 36;
export const NUM_CLASSES = 37;

export const INPUT_HEIGHT = 32;
export const INPUT_WIDTH = 128;

export const NORM_MEAN = 0.5;
export const NORM_STD = 0.5;
