"""augment v3 (ボツ): translate ±15% + scale 0.75-1.25 で fixed-head を破壊した版。

2026-06-01 実験で **isolated EM 48.1% → 1.9% の致命的退行**を起こしたため保存。
復帰時は augment.py を v1 spec に戻し、本ファイルは参照のみ。

## なぜ壊れたか (根本原因)

`fixed_head.py` の 12-head 固定長アーキテクチャは **位置固定の分類器**:
- 各位置 (0〜15) が独立した 13-class 分類を行う
- RNN / attention なし → 位置間の文脈伝播がない
- 「位置 p = 入力の p 番目の文字」という暗黙の契約に依存して学習される

augment v3 の translate ±15% (≈±19px、≈±2文字ぶん) は、この契約を破壊する:
- 訓練画像内で text が左右にランダムにシフトする
- 位置 0 (左端カラム) が見るのは 'E' (50%) / 黒パディング (25%) / '3' (25%)
- 結果、位置 0 は「'E' を予測しろ」を学習できず、最頻クラスの prior に逃げる

実際の予測例 (v3 訓練後):
- 入力: 様々な GT crop
- 出力: `E300MM000000`, `E300MM0008888`, `E300MM008881` ... 全部 prior 由来
- pattern_match は 98.1% (構造プリオールで通る) だが exact_match は 1.9%

## 教訓

**12-head fixed-length と translate augmentation は構造的に両立不可能**。
窓ズレ吸収は augment ではなく、runtime 側で window を再センタリングして
「位置 0 = 1文字目」契約を維持する経路で解決する (preprocess.ts:recenterBbox)。

CTC は alignment が動的なので translate に強いが、∅ class 相当の構造的 reject が
弱く、false positive 抑止と両立しにくい (本プロジェクトの reject 優先方針と相性悪)。
よって v3 → v1 復帰が選択された。

## v2 (別ボツ版) との違い

| 版 | 失敗原因 |
|---|---|
| v2 (`augment_v2_too_aggressive.py`) | blur/noise 強化が under-fit を招いた (データ量不足) |
| v3 (本ファイル) | translate/scale 拡大が fixed-head の位置契約を破壊 (構造的) |

v2 は「データ追加で復活可能」、v3 は「アーキテクチャ依存で根本的に NG」。
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
    """v3 augmentation: 窓ズレ耐性を組み込む (Phase 2c+ fix)。

    v1 → v3 で **translate と scale 範囲を拡大**:
    sliding-window 窓は GT text_bbox と IoU 0.47 程度しか重ならない (構造的天井)。
    クロップ内で text が中央からズレている、または scale が違う窓を訓練で経験させないと、
    isolated test で 48% EM 出る認識器でも E2E recall 0% に落ちる。

    v2 (boセ) は augment_v2_too_aggressive.py に保存済 (blur/noise 強化が under-fit を招いた)。
    v3 は **translate/scale だけ強化**して認識本体への影響は控えめ。
    """
    return A.Compose([
        # 幾何: 窓ズレ耐性を強化 (translate ±15%、scale 0.75-1.25)
        A.Affine(
            rotate=(-3, 3),
            scale=(0.75, 1.25),                 # 窓内 text サイズの揺らぎを学習
            translate_percent=(-0.15, 0.15),    # 窓中心からのズレを学習
            shear=(-2, 2),
            p=0.7,                              # ほぼ毎回適用
            mode=0,                             # cv2.BORDER_CONSTANT (= 0 padding)
        ),
        A.Perspective(scale=(0.01, 0.05), p=0.3),

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
