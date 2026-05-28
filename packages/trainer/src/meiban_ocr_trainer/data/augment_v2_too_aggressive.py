"""Albumentations による訓練時 augmentation。HANDOFF.md §4 Step 4 + v0 振り返り反映。

v2 (2026-05-27): v0 訓練の失敗パターン分析を元に強化版。
v0 結果: val_CER 3.85%、誤りは全て 1文字違い、特に 0/6/8/9 の数字混同が中心。
→ 数字の判別を学ばせるため、ボケ・ノイズ・解像度劣化・弾性変形を厚くする。

入力前提:
- numpy array (H, W, 3) BGR (cv2 ベース)
- 出力: torch.Tensor (1, INPUT_HEIGHT, INPUT_WIDTH) グレースケール、 [-1, 1] 正規化
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
    """v2 強化 augmentation。

    変更点 (v1→v2):
    - Affine: 回転 ±2→±3、scale 0.92-1.08 → 0.85-1.15、translate追加
    - ElasticTransform 新規追加 (alpha=20, sigma=4) — 文字形状の微妙な変動を学ばせる
    - CLAHE 追加 — コントラスト変動の幅を広げる
    - Blur: MotionBlur 5→9、GaussianBlur 3→(3,7)、Defocus 追加
    - ImageCompression: q=25-85 → q=15-80 (より低品質側)
    - Downscale: 0.5-0.9 → 0.3-0.85 (より極端な低解像度)
    - ISONoise intensity 0.1-0.5 → 0.2-0.8、GaussNoise 新規追加
    - CoarseDropout: 1-3 holes → 2-6 holes、サイズも拡大
    - Sharpen 追加 — blur と反対方向の劣化も学ばせる
    """
    return A.Compose([
        # 幾何変換 (やや強化)
        A.Affine(
            rotate=(-3, 3),
            scale=(0.85, 1.15),
            shear=(-3, 3),
            translate_percent=(-0.02, 0.02),
            p=0.5,
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.4),
        # Why: 文字形状の微変動 (筆画の太さ・角度) を増やし、0/6/8/9 のような形状近接文字の
        # 判別を強化する。alpha/sigma は大きすぎると判読不能になるので控えめ。
        A.ElasticTransform(alpha=20, sigma=4, p=0.25),

        # 照明 (やや強化)
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.RandomToneCurve(scale=0.2, p=0.2),

        # 質感劣化 (大幅強化)
        A.ImageCompression(quality_range=(15, 80), p=0.7),
        A.Downscale(
            scale_range=(0.3, 0.85),
            interpolation_pair={"upscale": 1, "downscale": 1},
            p=0.5,
        ),
        A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.2, 0.8), p=0.5),
        A.GaussNoise(std_range=(0.04, 0.15), p=0.4),

        # Blur (大幅強化 — 0/6/8/9 判別の鍵)
        A.OneOf([
            A.MotionBlur(blur_limit=9, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=0.6),

        # Sharpen (blur に対抗、デバイスのシャープニング処理を模擬)
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),

        # 反射・汚れ模擬 (強化)
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(2, 8),
            hole_width_range=(2, 8),
            p=0.4,
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
