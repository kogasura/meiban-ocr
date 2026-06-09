"""共有リサイズ。stretch(従来 A.Resize 相当) と letterbox(アスペクト保持 + 右下 0 埋め)を
1 関数に集約し、訓練(augment) と評価(diagnose_pipeline / eval_recognition) で**同一幾何**を保証する。

背景: モデル入力は 128x32 (AR=4) 固定だが、実シリアル crop は AR 5〜7 や、傾き crop で AR<4
が混在する。従来は全 crop を 128x32 に引き伸ばす(stretch)ため、正方形寄り(Q4=傾き)の crop が
水平方向に強く歪む。letterbox はアスペクトを保ったまま 128x32 の左上に収め、右・下を 0 で
パディングする(文字は左上アンカー=末尾桁の右に余白=右文脈を与える)。

注意: letterbox は AR>4 の横長 crop では高さが 32 未満に縮む(縦解像度を失う)トレードオフがある。
stretch との優劣は経験的なので、resize_mode を切替可能にして A/B 比較する。
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox_resize(
    img: np.ndarray,
    width: int,
    height: int,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """アスペクト比を保持して (height, width) のキャンバスに収め、右・下を pad_value で埋める。

    文字は左上アンカー。scale = min(width/w, height/h) で box 内に収める。
    """
    h, w = img.shape[:2]
    out_shape = (height, width) + img.shape[2:]
    if h == 0 or w == 0:
        return np.full(out_shape, pad_value, dtype=img.dtype)
    scale = min(width / w, height / h)
    nw = max(1, min(width, int(round(w * scale))))
    nh = max(1, min(height, int(round(h * scale))))
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.full(out_shape, pad_value, dtype=img.dtype)
    canvas[:nh, :nw] = resized
    return canvas


def resize_for_model(
    img: np.ndarray,
    width: int,
    height: int,
    mode: str = "stretch",
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """mode='stretch' は従来通り全面リサイズ、'letterbox' はアスペクト保持 + 右下 0 埋め。"""
    if mode == "letterbox":
        return letterbox_resize(img, width, height, pad_value=0, interpolation=interpolation)
    return cv2.resize(img, (width, height), interpolation=interpolation)
