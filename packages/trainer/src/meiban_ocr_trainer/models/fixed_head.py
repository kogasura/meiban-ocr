"""FixedHeadOCR: 12-position fixed-length OCR for Ericsson serials.

CRNN+CTC を置き換える Phase 2b アーキテクチャ。各位置 13 クラス (`0-9, E, M, ∅`)。
hallucination 問題を構造的に解決する設計。

設計判断:
- バックボーンは MobileNetV3-Small features[:10] を再利用 (TinyOCRModel と同じ調整)
- 32×128 入力 → backbone 後 (B, 96, 1, 32) → AdaptiveAvgPool で W: 32 → 12 に縮約
- 各 12 位置に独立した Linear(96 → 13) で分類
- BiGRU は採用しない (固定位置 = 各位置独立で学習可能、 BiGRU 0.5M params の節約)
  オプションで使いたい場合は use_rnn=True で復活可能

入力: (B, 1, 32, 128) グレースケール、[-1, 1]
出力: (B, 12, 13) logits

decode:
  各位置 argmax → 全位置が ∅ なら空文字 (reject)、それ以外は ∅ を除いた文字を連結
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from meiban_ocr_trainer.constants import FIXED_LENGTH, NUM_CLASSES_12H


def _build_backbone(pretrained: bool = True) -> tuple[nn.Sequential, int]:
    """MobileNetV3-Small features[:10] を取り出し、横方向 stride を抑制。

    TinyOCRModel と同じ backbone 構造を共有 (32×128 → 出力 (B, 96, 1, 32))。
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    base = mobilenet_v3_small(weights=weights)
    feats = list(base.features.children())[:10]
    for idx in (2, 4, 9):
        block = feats[idx]
        for m in block.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (2, 1)
                break
    return nn.Sequential(*feats), 96


class FixedHeadOCR(nn.Module):
    """12-position fixed-length head OCR。

    Input:  (B, 1, 32, 128)
    Output: (B, 12, 13)

    各位置は独立してクラス分類され、∅ (empty) クラスで「ここに文字無し」を表現する。
    全位置が ∅ なら出力は空文字 (= 構造的 reject)。
    """

    def __init__(
        self,
        num_positions: int = FIXED_LENGTH,
        num_classes: int = NUM_CLASSES_12H,
        use_rnn: bool = False,
        rnn_hidden: int = 64,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.num_classes = num_classes
        self.use_rnn = use_rnn

        self.backbone, backbone_out = _build_backbone(pretrained=pretrained)
        # 横方向の解像度を 32 → num_positions=16 に縮約 (32/16=2 で割り切れる)。
        # Why 16: ONNX export では AdaptiveAvgPool2d の出力サイズが入力サイズの
        # 約数である必要があり、Ericsson 文字数 12 では割り切れない (32/12 ≈ 2.67)。
        # 16 にすることで割り切れて export OK、Ericsson 12 文字は末尾 4 位置を ∅ で padding。
        self.pool = nn.AdaptiveAvgPool2d((1, num_positions))

        if use_rnn:
            # 副次的: 位置間の文脈を取り込みたい場合 (例: 'E' を見たら次は数字)
            self.rnn = nn.GRU(
                input_size=backbone_out,
                hidden_size=rnn_hidden,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0.0,
            )
            classifier_in = 2 * rnn_hidden
        else:
            self.rnn = None
            classifier_in = backbone_out

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(classifier_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # 1ch グレースケール → 3ch にレプリケーション (MobileNetV3 が要求)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)                  # (B, 96, 1, 32)
        feat = self.pool(feat)                   # (B, 96, 1, num_positions)
        feat = feat.squeeze(2)                   # (B, 96, num_positions)
        feat = feat.permute(0, 2, 1)             # (B, num_positions, 96)
        if self.rnn is not None:
            feat, _ = self.rnn(feat)             # (B, num_positions, 2H)
        feat = self.dropout(feat)
        return self.classifier(feat)             # (B, num_positions, num_classes)


__all__ = ["FixedHeadOCR"]
