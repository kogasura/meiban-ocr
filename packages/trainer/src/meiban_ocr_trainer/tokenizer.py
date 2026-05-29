"""文字列 ⇔ ラベル ID 列の変換。

2 つのトークナイザを提供:
- CTCTokenizer: CRNN+CTC (可変長、blank クラス) 用。TinyOCRModel と組み合わせる。
- FixedLengthTokenizer: 12-head 固定長 (各位置 13 クラス、∅ クラス) 用。FixedHeadOCR と組み合わせる。
"""

from __future__ import annotations

import torch

from meiban_ocr_trainer.constants import (
    BLANK_IDX,
    CHARSET,
    CHARSET_12H,
    EMPTY_IDX,
    FIXED_LENGTH,
    NUM_CLASSES,
    NUM_CLASSES_12H,
)


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


class FixedLengthTokenizer:
    """12-position fixed-length tokenizer (Ericsson serial 専用、charset 12 文字 + ∅)。

    encode: 文字列 → 長さ FIXED_LENGTH の固定 ID 列
        - text="" or 空文字: 全位置が EMPTY_IDX (= reject 目標)
        - text="E300MM000001": 各位置の char→idx + 残りは EMPTY_IDX
        - text 長 > FIXED_LENGTH: ValueError
        - char が CHARSET_12H 外: ValueError

    decode: (B, L, C) logits → 文字列リスト
        - 各位置 argmax → EMPTY_IDX なら出力に含めない
        - 全位置が EMPTY_IDX なら空文字 (構造的 reject)
    """

    def __init__(
        self,
        charset: str = CHARSET_12H,
        empty_idx: int = EMPTY_IDX,
        fixed_length: int = FIXED_LENGTH,
    ) -> None:
        if empty_idx != len(charset):
            raise ValueError(
                f"empty_idx ({empty_idx}) must equal len(charset) ({len(charset)})"
            )
        self.charset = charset
        self.empty_idx = empty_idx
        self.fixed_length = fixed_length
        self.num_classes = empty_idx + 1
        self._char_to_idx = {c: i for i, c in enumerate(charset)}

    def encode(self, text: str) -> list[int]:
        """文字列を長さ fixed_length のラベル列に変換 (足りない位置は EMPTY_IDX)。"""
        if len(text) > self.fixed_length:
            raise ValueError(
                f"text length {len(text)} exceeds fixed_length {self.fixed_length}"
            )
        ids: list[int] = []
        for c in text:
            try:
                ids.append(self._char_to_idx[c])
            except KeyError as e:
                raise ValueError(
                    f"character {e.args[0]!r} not in CHARSET_12H ({self.charset!r})"
                ) from e
        # ∅ で右パディング
        ids += [self.empty_idx] * (self.fixed_length - len(ids))
        return ids

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """バッチエンコード。Returns (B, fixed_length) long tensor。"""
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def decode(self, logits: torch.Tensor) -> list[str]:
        """Fixed-length decode: 各位置 argmax → ∅ を除いた文字を連結。

        Args:
            logits: (B, L, C). L = fixed_length, C = num_classes.

        Returns:
            デコードされた文字列のリスト (length B)。全位置 ∅ なら空文字。
        """
        results = self.decode_with_conf(logits)
        return [text for text, _ in results]

    def decode_with_conf(
        self, logits: torch.Tensor,
    ) -> list[tuple[str, float]]:
        """Fixed-length decode + per-sample aggregated confidence。

        confidence の集約方式 (Phase 2c, 業界ベスト + 我々の要件):

            confidence = min(geomean(top1), min(top1))   全 12 位置を対象 (∅ 含む)

        - geomean(top1): EasyOCR と同等。全体的に「そこそこ高い」を要求 = 独立性仮定での
          全体正解確率 (= prod(top1)) を 12 乗根で位置あたりに正規化したもの。
        - min(top1): 弱点位置の確率。「1 文字でも怪しければ全体を落とす」要件を直接表現。
        - **両者の min を取る**: 厳しい側を採用、false-accept を最優先で避ける方針と整合。

        ∅ 位置も集約対象に含める (旧版は除外していた): ∅ が低 prob で出るのは混乱の証拠、
        ∅ が高 prob で出るのは構造異常、いずれも reject 方向に倒したい。

        Args:
            logits: (B, L, C) raw logits.

        Returns:
            [(decoded_text, confidence), ...] (length B)。
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D, got shape {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"last dim must be num_classes={self.num_classes}, "
                f"got {logits.shape[-1]}"
            )
        if logits.shape[1] != self.fixed_length:
            raise ValueError(
                f"sequence length must be fixed_length={self.fixed_length}, "
                f"got {logits.shape[1]}"
            )

        probs = torch.softmax(logits, dim=-1).cpu()
        best_idx = probs.argmax(dim=-1)             # (B, L)
        best_prob = probs.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)  # (B, L)

        out: list[tuple[str, float]] = []
        for b in range(probs.shape[0]):
            seq = best_idx[b].tolist()
            # text: 非 ∅ 位置だけ連結 (∅ は文字なし)
            chars = [self.charset[idx] for idx in seq if idx != self.empty_idx]

            # confidence: 全 12 位置の top1 prob から min(geomean, min)
            top1 = best_prob[b]                     # (L,)
            log_top1 = torch.log(top1.clamp_min(1e-9))
            geomean = float(torch.exp(log_top1.mean()))
            min_top1 = float(top1.min())
            conf = min(geomean, min_top1)

            out.append(("".join(chars), conf))
        return out

    def decode_detailed(
        self, logits: torch.Tensor,
    ) -> list[dict]:
        """全位置の詳細情報を返す版 (runtime API 設計用)。

        Returns:
            [
              {
                "text": "E300MM000001",
                "confidence": 0.92,           # min(geomean, min)
                "geomean": 0.95,              # 参考
                "min_top1": 0.92,             # 参考
                "min_margin": 0.85,           # top1 - top2 の最小
                "per_position": [
                  {"char": "E" or None for ∅, "top1": 0.99, "top2_char": "∅",
                   "top2": 0.01, "margin": 0.98},
                  ...
                ],
              },
              ...
            ]

        この豊富な出力を integration 側に渡すと、用途に応じた gate 設計が可能になる。
        """
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D, got shape {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=-1).cpu()
        top2_vals, top2_idx = probs.topk(2, dim=-1)  # both (B, L, 2)

        results: list[dict] = []
        for b in range(probs.shape[0]):
            chars: list[str] = []
            per_pos: list[dict] = []
            top1_list: list[float] = []
            margin_list: list[float] = []
            for t in range(self.fixed_length):
                idx1 = int(top2_idx[b, t, 0])
                idx2 = int(top2_idx[b, t, 1])
                p1 = float(top2_vals[b, t, 0])
                p2 = float(top2_vals[b, t, 1])
                if idx1 != self.empty_idx:
                    chars.append(self.charset[idx1])
                per_pos.append({
                    "char": self.charset[idx1] if idx1 != self.empty_idx else None,
                    "top1": p1,
                    "top2_char": (
                        self.charset[idx2] if idx2 != self.empty_idx else None
                    ),
                    "top2": p2,
                    "margin": p1 - p2,
                })
                top1_list.append(p1)
                margin_list.append(p1 - p2)

            import math
            log_top1 = [math.log(max(p, 1e-9)) for p in top1_list]
            geomean = math.exp(sum(log_top1) / len(log_top1))
            min_top1 = min(top1_list)
            min_margin = min(margin_list)
            conf = min(geomean, min_top1)

            results.append({
                "text": "".join(chars),
                "confidence": conf,
                "geomean": geomean,
                "min_top1": min_top1,
                "min_margin": min_margin,
                "per_position": per_pos,
            })
        return results


__all__ = [
    "CTCTokenizer",
    "FixedLengthTokenizer",
    "BLANK_IDX",
    "CHARSET",
    "NUM_CLASSES",
    "CHARSET_12H",
    "EMPTY_IDX",
    "NUM_CLASSES_12H",
    "FIXED_LENGTH",
]
