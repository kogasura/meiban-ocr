"""Albumentations による訓練時 augmentation。HANDOFF.md §4 Step 4 を実装。

質感劣化系を厚めに、幾何は控えめ (撮影は真上付近・固定照明と想定)。

入力前提:
- numpy array (H, W, 3) BGR (cv2 ベース)
- 出力: torch.Tensor (1, INPUT_HEIGHT, INPUT_WIDTH) グレースケール、 [-1, 1] 正規化

Version history:
- v1 (2026-05-27 初版): val_CER 3.85%、val_EM 53.8% を達成 (38 real + 1900 replaced で訓練)
- v2 (2026-05-27、ボツ): blur/noise/Elastic を厚くしたが、1938サンプル規模では under-fit。
  train_loss が 0.62 で頭打ち、val_CER も 0.0641 に悪化。v2 コードは
  `augment_v2_too_aggressive.py` に保存。データ量を増やしてから再挑戦の予定。

Why this is v1 (緩め): 訓練データが限定的 (38 real + 1900 synthetic) のため、強い
augmentation は under-fit を引き起こす。データ量に対し augment 強度のバランスが重要。
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import torch

from meiban_ocr_trainer.constants import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NORM_MEAN,
    NORM_STD,
)


def build_train_transform() -> A.Compose:
    """v1 (緩め) augmentation。"""
    return A.Compose([
        # 幾何 (真上付近想定なので控えめ)
        A.Affine(rotate=(-2, 2), scale=(0.92, 1.08), shear=(-2, 2), p=0.4),
        A.Perspective(scale=(0.01, 0.04), p=0.3),

        # 照明
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # 質感劣化
        A.ImageCompression(quality_range=(25, 85), p=0.6),
        A.Downscale(
            scale_range=(0.5, 0.9),
            interpolation_pair={"upscale": 1, "downscale": 1},
            p=0.3,
        ),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.4),

        # 反射・汚れ模擬
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(2, 6),
            hole_width_range=(2, 6),
            p=0.3,
        ),

        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH, interpolation=1),
    ])


def build_eval_transform() -> A.Compose:
    """評価時 transform: resize のみ。"""
    return A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH, interpolation=1),
    ])


def to_model_tensor(arr_bgr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) BGR uint8 → (1, H, W) float32 [-1, 1] グレースケール。"""
    import cv2

    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    arr = gray.astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN[0]) / NORM_STD[0]
    return torch.from_numpy(arr).unsqueeze(0)
