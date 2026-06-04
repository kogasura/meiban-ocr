"""CRNNPretrained: clovaai None-VGG-BiLSTM-CTC backbone を流用した CRNN。

clovaai/deep-text-recognition-benchmark の公式 pretrained weight を使う:
  - 訓練データ: MJSynth (800万) + SynthText (600万) = 1400万 text crop
  - License: Apache 2.0
  - 文字集合 (clovaai): 0-9 + **a-z** (小文字 36)
  - 我々の charset: 0-9 + **A-Z** (大文字 36)

Why this design:
  我々の MobileNetV3-Small + Bi-GRU (TinyOCRModel) は ImageNet pretrain のみで OCR 専用
  事前学習がない。 結果 per-char accuracy 88% で頭打ち (digit 識別能力が backbone レベルで
  獲得されていない)。 clovaai backbone は 1400万 text crop で digit 認識を獲得済 → そこ
  からの fine-tune で per-char 99%+ を狙う。

実装:
  - VGG_FeatureExtractor + 2x BidirectionalLSTM (clovaai と同 arch)
  - Prediction layer は **reinit**: clovaai は小文字 charset で訓練、 我々の大文字 charset
    と互換性なし → Linear(256→37) を新規初期化 + fine-tune で再学習
  - Backbone + BiLSTM weight は state_dict 経由で load (transfer)

入力: (B, 1, 32, 128) グレースケール [-1, 1] (我々の前処理と一致)
出力: (B, T, 37) logits、 T ≈ 31 (32×128 → VGG → 1×31)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _build_vgg_feature_extractor(input_channel: int = 1, output_channel: int = 512) -> nn.Sequential:
    """clovaai/modules/feature_extraction.py:VGG_FeatureExtractor と完全同一。

    重要: 各 nn.Module の **インデックス**が clovaai と一致する必要あり (state_dict key
    `FeatureExtraction.ConvNet.0`, `.3`, `.6` 等を load するため)。 ReLU の inplace=True と
    MaxPool の (k, s) も合わせる。
    """
    # output channel は clovaai 内部で output_channel / [8, 4, 2, 1] = [64, 128, 256, 512]
    return nn.Sequential(
        # index 0-2: Conv + ReLU + MaxPool
        nn.Conv2d(input_channel, 64, 3, 1, 1),    # 0
        nn.ReLU(True),                            # 1
        nn.MaxPool2d(2, 2),                       # 2
        # index 3-5
        nn.Conv2d(64, 128, 3, 1, 1),              # 3
        nn.ReLU(True),                            # 4
        nn.MaxPool2d(2, 2),                       # 5
        # index 6-10
        nn.Conv2d(128, 256, 3, 1, 1),             # 6
        nn.ReLU(True),                            # 7
        nn.Conv2d(256, 256, 3, 1, 1),             # 8
        nn.ReLU(True),                            # 9
        nn.MaxPool2d((2, 1), (2, 1)),             # 10
        # index 11-16 (BatchNorm)
        nn.Conv2d(256, 512, 3, 1, 1, bias=False), # 11
        nn.BatchNorm2d(512),                       # 12
        nn.ReLU(True),                            # 13
        nn.Conv2d(512, 512, 3, 1, 1, bias=False), # 14
        nn.BatchNorm2d(512),                       # 15
        nn.ReLU(True),                            # 16
        nn.MaxPool2d((2, 1), (2, 1)),             # 17
        # index 18: 最終 Conv
        nn.Conv2d(512, 512, 2, 1, 0),             # 18
        nn.ReLU(True),                            # 19
    )


class BidirectionalLSTM(nn.Module):
    """clovaai/modules/sequence_modeling.py:BidirectionalLSTM。

    LSTM (bidirectional, 1 layer) + Linear で次元調整。 nn.Module 構造を一致させる。
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class CRNNPretrained(nn.Module):
    """clovaai None-VGG-BiLSTM-CTC アーキ + reinitiated Prediction head。

    fine-tune 戦略:
      - FeatureExtraction (backbone): LR 低 (1e-5)、 pretrain 情報温存
      - SequenceModeling (BiLSTM): LR 中 (1e-4)、 軽い再調整
      - Prediction (新規 init): LR 高 (1e-3)、 大文字 charset への適応
    """

    INPUT_HEIGHT = 32
    INPUT_WIDTH = 128
    PRETRAINED_NUM_CLASSES = 37  # clovaai (36 lowercase + blank)
    OUR_NUM_CLASSES = 37  # 36 uppercase + blank

    def __init__(self, num_classes: int = OUR_NUM_CLASSES, hidden_size: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # clovaai 互換のフィールド名 (state_dict load を素直に通す)
        self.FeatureExtraction = nn.Module()
        self.FeatureExtraction.ConvNet = _build_vgg_feature_extractor(input_channel=1, output_channel=512)

        # AdaptiveAvgPool (clovaai) は (None, 1) で permute 後の最後の dim を 1 にするが、
        # 我々の入力 32×128 では VGG 出力が常に (B, 512, 1, 31) → permute (B, 31, 512, 1)
        # → 最後の dim 既に 1 で AdaptiveAvgPool は **no-op**。
        # ONNX export で `AdaptiveAvgPool2d((None, 1))` は output_size 非定数として
        # サポート外 → モジュールごと省略し、 forward で squeeze(-1) 相当を直接行う。
        # 重みなしモジュールなので state_dict load にも影響なし。

        # Sequence Modeling: 2 layer BiLSTM
        # clovaai は SequenceModeling_input = output_channel = 512、 hidden_size = 256
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )

        # Prediction: Linear(hidden, num_class)
        # **これだけ reinit する**。 clovaai は小文字 36 + blank、 我々は大文字 36 + blank
        # で 出力 index の意味が違うため。 fine-tune 時に新しい mapping を学ぶ。
        self.Prediction = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力 (B, 1, 32, 128) → 出力 (B, T=31, num_classes)。

        AdaptiveAvgPool は no-op として省略 (ONNX 互換のため、 forward 内で squeeze)。
        """
        # VGG features: 32×128 入力で出力 (B, 512, 1, 31)
        visual = self.FeatureExtraction.ConvNet(x)
        # squeeze H=1 → (B, 512, 31)、 permute → (B, 31, 512)
        # (clovaai は permute → (B, 31, 512, 1) → pool → squeeze(3) で同じ結果)
        visual = visual.squeeze(2).permute(0, 2, 1)  # (B, 31, 512)

        # BiLSTM × 2
        contextual = self.SequenceModeling(visual)  # (B, 31, hidden_size)

        # Prediction
        logits = self.Prediction(contextual)  # (B, 31, num_classes)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        weight_path: str | Path,
        num_classes: int = OUR_NUM_CLASSES,
        hidden_size: int = 256,
    ) -> "CRNNPretrained":
        """clovaai pretrained weight を load して新規 model を返す。

        手順:
          1. `torch.load(weight_path)` で state_dict を取得 (key は `module.<...>.<param>` 形式)
          2. `module.` prefix を除去
          3. `module.Prediction.*` keys を **drop** (=新しい大文字 charset 用に reinit)
          4. `load_state_dict(..., strict=False)` で残りを load
        """
        model = cls(num_classes=num_classes, hidden_size=hidden_size)
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        # ckpt は {'state_dict': ...} 形式とは限らない (clovaai は直接 state_dict 形式)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            state = ckpt.state_dict()

        # "module." prefix 除去
        cleaned = {}
        for k, v in state.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            cleaned[new_k] = v

        # Prediction layer を除外 (新しい charset で reinit)
        # FeatureExtraction.ConvNet.* + SequenceModeling.* のみ採用
        for k in list(cleaned.keys()):
            if k.startswith("Prediction."):
                del cleaned[k]

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if unexpected:
            print(f"[CRNNPretrained.from_pretrained] WARNING: unexpected keys: {unexpected}")
        # missing は Prediction.{weight,bias} のみが期待値
        expected_missing = {"Prediction.weight", "Prediction.bias"}
        actual_missing = set(missing)
        if not actual_missing.issubset(expected_missing) and actual_missing != expected_missing:
            extra_missing = actual_missing - expected_missing
            if extra_missing:
                print(f"[CRNNPretrained.from_pretrained] WARNING: missing keys (extra beyond Prediction): {extra_missing}")
        return model

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        rnn_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """fine-tune 用の layered LR param groups を返す。

        - backbone (FeatureExtraction): 低 LR、 pretrain 情報温存
        - RNN (SequenceModeling): 中 LR、 シーケンス文脈の微調整
        - head (Prediction): 高 LR、 新規 charset への高速適応
        """
        return [
            {
                "params": [p for p in self.FeatureExtraction.parameters() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": [p for p in self.SequenceModeling.parameters() if p.requires_grad],
                "lr": rnn_lr,
                "weight_decay": weight_decay,
                "name": "rnn",
            },
            {
                "params": [p for p in self.Prediction.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def freeze_backbone(self, freeze: bool = True) -> None:
        """warmup 用 backbone freeze。"""
        for p in self.FeatureExtraction.parameters():
            p.requires_grad = not freeze


__all__ = ["CRNNPretrained", "BidirectionalLSTM"]
