"""CTC tokenizer: 文字列 ⇔ ラベルID列。"""

from __future__ import annotations

import torch

from meiban_ocr_trainer.constants import BLANK_IDX, CHARSET, NUM_CLASSES


class CTCTokenizer:
    """36文字 (0-9A-Z) + CTC blank の双方向変換。"""

    def __init__(self, charset: str = CHARSET, blank_idx: int = BLANK_IDX) -> None:
        if blank_idx != len(charset):
            # Why: ランタイム側 (TS) でも blank を charset 直後の index 固定で扱うため、
            # ここで強制してずれを早期検出する。
            raise ValueError(
                f"blank_idx ({blank_idx}) must equal len(charset) ({len(charset)})"
            )
        self.charset = charset
        self.blank_idx = blank_idx
        self.num_classes = blank_idx + 1
        self._char_to_idx = {c: i for i, c in enumerate(charset)}

    def encode(self, text: str) -> list[int]:
        """文字列をラベル列に変換。CHARSET にない文字は ValueError。"""
        try:
            return [self._char_to_idx[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in CHARSET") from e

    def encode_batch(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CTCLoss 入力形式に整形。

        Returns:
            targets: 1次元連結テンソル (sum(target_lengths),)
            target_lengths: 各サンプルのラベル長 (B,)
        """
        lengths = [len(t) for t in texts]
        flat: list[int] = []
        for t in texts:
            flat.extend(self.encode(t))
        return (
            torch.tensor(flat, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def greedy_decode(self, logits: torch.Tensor) -> list[str]:
        """Greedy CTC decode.

        Args:
            logits: (B, T, C) または (T, B, C)。前者を想定。

        Returns:
            デコードされた文字列のリスト (length B)。
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D, got shape {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"last dim must be num_classes={self.num_classes}, "
                f"got {logits.shape[-1]}"
            )
        # (B, T, C) → (B, T)
        best = logits.argmax(dim=-1).cpu().tolist()
        out: list[str] = []
        for seq in best:
            prev = -1
            chars: list[str] = []
            for idx in seq:
                if idx != prev and idx != self.blank_idx:
                    chars.append(self.charset[idx])
                prev = idx
            out.append("".join(chars))
        return out


__all__ = ["CTCTokenizer", "BLANK_IDX", "CHARSET", "NUM_CLASSES"]
