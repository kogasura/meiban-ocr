"""TinyOCRModel: MobileNetV3-Small + Bi-GRU + CTC head.

HANDOFF.md §3 のスケッチ + HANDOFF_ADDENDUM.md §1, §2 を実装。仕様からの逸脱2点:

Why (stride 調整): HANDOFF.md のサンプルは MobileNetV3-Small `features[:10]` をそのまま
使うが、それだと 32x128 入力に対し縦横とも 5回 stride=2 が適用されて出力が 1x4 になり、
CTC の T=4 では L=12 文字を出力できない (CTC は概ね T >= 2L-1 が必要)。そのため後段ブロックの
**横** stride を 1 に潰し、T を確保する。

Why (ImageNet pretrained, ADDENDUM §1 バグ修正): バックボーンは ImageNet 事前学習を継承する。
ゼロから学習させると小データで収束しにくく汎化性能が落ちる。`pretrained=False` も渡せるよう
にしてあるのは、ネット非依存のテスト用 (重みのダウンロードを避ける)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from meiban_ocr_trainer.constants import NUM_CLASSES


def _build_backbone(pretrained: bool = True) -> tuple[nn.Sequential, int]:
    """MobileNetV3-Small の前半 (features[:10]) を取り出し、horizontal stride を調整。

    Args:
        pretrained: True なら ImageNet1K_V1 重みをロード (HANDOFF_ADDENDUM.md §1)。

    Returns:
        (backbone, out_channels)
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    base = mobilenet_v3_small(weights=weights)
    feats = list(base.features.children())[:10]
    # Why: features[2], [4], [9] の stride=(2,2) のうち横方向を 1 に変えると、
    # 32x128 入力 → 出力 1x32 (channels=96) になる (T=32, L=12 で十分)。
    # 該当ブロックの最初の depthwise/conv の stride を (2,1) に書き換える。
    for idx in (2, 4, 9):
        block = feats[idx]
        for m in block.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (2, 1)
                # padding は維持。kernel_size に応じて出力サイズが整数になる前提。
                break  # ブロック内の最初の downsampling Conv のみ書き換える
    return nn.Sequential(*feats), 96


class TinyOCRModel(nn.Module):
    """軽量CRNN。

    Input:  (B, 1, 32, 128) グレースケール、[-1, 1] (NORM_MEAN=0.5/NORM_STD=0.5) を想定。
    Output: (B, T, num_classes) logits。T はバックボーンの出力幅 (調整後で T=32)。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Why: MobileNetV3 は 3ch 入力前提。1ch グレースケールは channel-replication で渡す。
        # 入力チャンネルは forward で expand する (Conv の重みを書き換えるより副作用が少ない)。
        self.backbone, backbone_out = _build_backbone(pretrained=pretrained)
        # Why: backbone は 32x128 入力に対し (B, 96, 1, 32) を出力するため H 方向は
        # 既に 1。元仕様 (HANDOFF.md §3) は `nn.AdaptiveAvgPool2d((1, None))` だが、
        # ONNX export では output_size に動的次元を含むと未サポートになる。
        # forward 内で `feat.mean(dim=2)` に置き換える (H=1 想定で意味は同じ。
        # 別解像度入力時は H>1 になり平均で集約される、これも妥当な挙動)。
        # パラメータ無し演算なので state_dict 互換は保たれる。
        self.rnn = nn.GRU(
            input_size=backbone_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)            # (B, C, H', W')
        feat = feat.mean(dim=2)            # (B, C, W')  H 方向を平均で集約 (H=1 でも安全)
        feat = feat.permute(0, 2, 1)       # (B, W', C)  W' = T
        out, _ = self.rnn(feat)            # (B, T, 2H)
        return self.classifier(out)        # (B, T, num_classes)


__all__ = ["TinyOCRModel"]
