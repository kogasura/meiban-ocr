"""CRNNAttn: CTC encoder (CRNNPretrained と同一) + Bahdanau attention decoder。

Why (pos11 = CTC 末尾の構造弱点):
  CTC は各列の独立分類 + アライメント縮約のため、(a) 末尾文字に使える列が最少、
  (b) 既読文脈を使えない、(c) 出力長の制御が blank 頼み。clean test の誤りの大半
  (368/431) と本番誤読の 95% が pos10/11 に集中する。
  attention decoder は 1 文字ずつ自己回帰生成し、毎ステップ「どの列を見るか」を
  動的に再計算する。末尾でも注視は対等、既読文脈 (E325MM5007…) を条件に使え、
  EOS で長さを明示的に止める。

fixed-head の失敗 (位置剛直で回転と非両立) と違い、attention は CTC と同じく
位置を動的に追従するため回転 robust 性と両立する。

設計:
  - encoder = CRNNPretrained と同一構造 (FeatureExtraction + SequenceModeling)。
    **訓練済み CTC checkpoint (例: runs/cr_replaced/best.pt) から warm start** し、
    ドメイン適応済みの特徴を再利用する。
  - CTC head (Prediction) は補助損失として残す (アライメント学習の錨 + 退行検知)。
  - decoder = Embedding + Bahdanau attention + GRUCell + Linear。
    最大 MAX_DECODE_STEPS(13) ステップ。負例は step0 で EOS (構造的 reject)。
  - 推論 greedy はループ固定回数なので ONNX へ静的展開で export 可能。

入力: (B, 1, 32, 128) → ctc_logits (B, 31, 37), attn_logits (B, 13, 37)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from meiban_ocr_trainer.constants import (
    MAX_DECODE_STEPS,
    NUM_CLASSES,
    NUM_CLASSES_ATTN,
    NUM_EMBEDDINGS_ATTN,
    SOS_IDX,
)
from meiban_ocr_trainer.models.crnn_pretrained import (
    BidirectionalLSTM,
    _build_vgg_feature_extractor,
)


class BahdanauAttention(nn.Module):
    """additive attention: score = v^T tanh(W_h h + W_e enc)。"""

    def __init__(self, enc_dim: int, hidden_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_hid = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, enc: torch.Tensor, enc_proj: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """enc (B,T,E), enc_proj (B,T,A) [事前計算], hidden (B,H) → context (B,E)。"""
        score = self.v(torch.tanh(enc_proj + self.w_hid(hidden).unsqueeze(1)))  # (B,T,1)
        alpha = torch.softmax(score, dim=1)
        return (alpha * enc).sum(dim=1)

    def project_encoder(self, enc: torch.Tensor) -> torch.Tensor:
        return self.w_enc(enc)


class CRNNAttn(nn.Module):
    """CTC encoder + attention decoder (joint)。"""

    def __init__(
        self,
        num_ctc_classes: int = NUM_CLASSES,
        hidden_size: int = 256,
        embed_dim: int = 64,
        attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # ---- encoder: CRNNPretrained と同一のフィールド名 (warm start のため) ----
        self.FeatureExtraction = nn.Module()
        self.FeatureExtraction.ConvNet = _build_vgg_feature_extractor(1, 512)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        # 補助 CTC head (warm start で訓練済みの重みも載る)
        self.Prediction = nn.Linear(hidden_size, num_ctc_classes)

        # ---- decoder ----
        self.embedding = nn.Embedding(NUM_EMBEDDINGS_ATTN, embed_dim)
        self.attention = BahdanauAttention(hidden_size, hidden_size, attn_dim)
        self.decoder_cell = nn.GRUCell(embed_dim + hidden_size, hidden_size)
        self.init_hidden = nn.Linear(hidden_size, hidden_size)
        self.generator = nn.Linear(hidden_size * 2, NUM_CLASSES_ATTN)

    # ---- encoder ----
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.FeatureExtraction.ConvNet(x)        # (B, 512, 1, 31)
        visual = visual.squeeze(2).permute(0, 2, 1)       # (B, 31, 512)
        return self.SequenceModeling(visual)              # (B, 31, hidden)

    # ---- decoder 1 step ----
    def _decode_step(
        self,
        prev_ids: torch.Tensor,
        hidden: torch.Tensor,
        enc: torch.Tensor,
        enc_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.attention(enc, enc_proj, hidden)                    # (B, H)
        emb = self.embedding(prev_ids)                                 # (B, E)
        hidden = self.decoder_cell(torch.cat([emb, ctx], dim=1), hidden)
        logits = self.generator(torch.cat([hidden, ctx], dim=1))       # (B, C)
        return logits, hidden

    def forward(
        self, x: torch.Tensor, teacher_inputs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (ctc_logits (B,T,37), attn_logits (B,S,37))。

        teacher_inputs (B, S) があれば teacher forcing、なければ greedy 自己回帰。
        greedy は固定 S 回ループ (EOS 後のステップは無意味な logits だが decode 側で
        EOS 打ち切りされるため無害。固定回数なので ONNX 静的展開可)。
        """
        enc = self.encode(x)
        ctc_logits = self.Prediction(enc)

        enc_proj = self.attention.project_encoder(enc)
        b = x.size(0)
        hidden = torch.tanh(self.init_hidden(enc.mean(dim=1)))
        step_logits: list[torch.Tensor] = []
        if teacher_inputs is not None:
            for s in range(teacher_inputs.size(1)):
                logits, hidden = self._decode_step(
                    teacher_inputs[:, s], hidden, enc, enc_proj
                )
                step_logits.append(logits)
        else:
            prev = torch.full((b,), SOS_IDX, dtype=torch.long, device=x.device)
            for _ in range(MAX_DECODE_STEPS):
                logits, hidden = self._decode_step(prev, hidden, enc, enc_proj)
                step_logits.append(logits)
                prev = logits.argmax(dim=-1)
        return ctc_logits, torch.stack(step_logits, dim=1)

    # ---- warm start ----
    @classmethod
    def from_ctc_checkpoint(
        cls, ckpt_path: str | Path, hidden_size: int = 256, **kw
    ) -> "CRNNAttn":
        """訓練済み CTC checkpoint (CRNNPretrained) から encoder + CTC head を継承。

        decoder (embedding/attention/decoder_cell/init_hidden/generator) のみ新規。
        """
        model = cls(hidden_size=hidden_size, **kw)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[CRNNAttn.from_ctc_checkpoint] WARNING unexpected: {unexpected}")
        decoder_prefixes = (
            "embedding.", "attention.", "decoder_cell.", "init_hidden.", "generator.",
        )
        extra = [m for m in missing if not m.startswith(decoder_prefixes)]
        if extra:
            print(f"[CRNNAttn.from_ctc_checkpoint] WARNING missing (non-decoder): {extra}")
        return model

    # ---- 訓練ユーティリティ ----
    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        rnn_lr: float = 5e-5,
        ctc_head_lr: float = 1e-4,
        decoder_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """warm start 前提の layered LR (encoder は温存、decoder は高速学習)。"""
        dec_params = (
            list(self.embedding.parameters())
            + list(self.attention.parameters())
            + list(self.decoder_cell.parameters())
            + list(self.init_hidden.parameters())
            + list(self.generator.parameters())
        )
        return [
            {"params": list(self.FeatureExtraction.parameters()), "lr": backbone_lr,
             "weight_decay": weight_decay, "name": "backbone"},
            {"params": list(self.SequenceModeling.parameters()), "lr": rnn_lr,
             "weight_decay": weight_decay, "name": "rnn"},
            {"params": list(self.Prediction.parameters()), "lr": ctc_head_lr,
             "weight_decay": weight_decay, "name": "ctc_head"},
            {"params": dec_params, "lr": decoder_lr,
             "weight_decay": weight_decay, "name": "decoder"},
        ]

    def freeze_encoder(self, freeze: bool = True) -> None:
        """decoder warmup 用: encoder (backbone+RNN+CTC head) を凍結。"""
        for m in (self.FeatureExtraction, self.SequenceModeling, self.Prediction):
            for p in m.parameters():
                p.requires_grad = not freeze


__all__ = ["CRNNAttn", "BahdanauAttention"]
