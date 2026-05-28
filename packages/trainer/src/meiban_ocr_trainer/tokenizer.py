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
        results = self.greedy_decode_with_conf(logits)
        return [text for text, _ in results]

    def greedy_decode_with_conf(
        self, logits: torch.Tensor,
    ) -> list[tuple[str, float]]:
        """Greedy CTC decode + per-sample mean confidence.

        confidence は「出力文字に貢献した non-blank timestep の softmax 最大確率の平均」。
        空出力 (CTC が全 blank を出した) の場合は **全 timestep の (1 - blank prob) 平均**
        を「reject 確信度」として返す (1 に近いほど自信を持って空を選んだ)。
        スコア範囲: [0, 1].

        Args:
            logits: (B, T, C). softmax 前の raw logits を想定。

        Returns:
            [(decoded_text, mean_confidence), ...] の長さ B。
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D, got shape {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"last dim must be num_classes={self.num_classes}, "
                f"got {logits.shape[-1]}"
            )
        probs = torch.softmax(logits, dim=-1).cpu()  # (B, T, C)
        best_idx = probs.argmax(dim=-1)  # (B, T)
        best_prob = probs.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

        out: list[tuple[str, float]] = []
        for b in range(probs.shape[0]):
            seq = best_idx[b].tolist()
            confs = best_prob[b].tolist()
            blank_probs = probs[b, :, self.blank_idx].tolist()

            prev = -1
            chars: list[str] = []
            char_confs: list[float] = []
            for t, idx in enumerate(seq):
                if idx != prev and idx != self.blank_idx:
                    chars.append(self.charset[idx])
                    char_confs.append(confs[t])
                prev = idx

            if chars:
                conf = sum(char_confs) / len(char_confs)
            else:
                # 全 blank → reject 確信度 (= blank prob の平均)
                conf = sum(blank_probs) / len(blank_probs)
            out.append(("".join(chars), float(conf)))
        return out


__all__ = ["CTCTokenizer", "BLANK_IDX", "CHARSET", "NUM_CLASSES"]
