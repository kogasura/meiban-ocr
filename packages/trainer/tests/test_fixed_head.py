"""FixedHeadOCR + FixedLengthTokenizer のテスト (Phase 2b)。"""

from __future__ import annotations

import pytest
import torch

from meiban_ocr_trainer.constants import (
    CHARSET_12H,
    EMPTY_IDX,
    FIXED_LENGTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES_12H,
)
from meiban_ocr_trainer.models import FixedHeadOCR
from meiban_ocr_trainer.tokenizer import FixedLengthTokenizer


# ----- Tokenizer -----

def test_charset_12h_is_expected() -> None:
    assert CHARSET_12H == "0123456789EM"
    assert EMPTY_IDX == 12
    assert NUM_CLASSES_12H == 13
    assert FIXED_LENGTH == 12


def test_encode_ericsson_serial() -> None:
    tok = FixedLengthTokenizer()
    ids = tok.encode("E300MM000001")
    # E=10, 3=3, 0=0, 0=0, M=11, M=11, 0=0, 0=0, 0=0, 0=0, 0=0, 1=1
    assert ids == [10, 3, 0, 0, 11, 11, 0, 0, 0, 0, 0, 1]
    assert len(ids) == FIXED_LENGTH


def test_encode_empty_string_pads_with_empty() -> None:
    """空文字 → 全位置 EMPTY_IDX (negative の reject 目標)。"""
    tok = FixedLengthTokenizer()
    ids = tok.encode("")
    assert ids == [EMPTY_IDX] * FIXED_LENGTH


def test_encode_short_string_pads_with_empty() -> None:
    """短い文字列 → 残り位置は EMPTY_IDX。"""
    tok = FixedLengthTokenizer()
    ids = tok.encode("E30")
    expected = [10, 3, 0] + [EMPTY_IDX] * (FIXED_LENGTH - 3)
    assert ids == expected


def test_encode_rejects_too_long() -> None:
    tok = FixedLengthTokenizer()
    with pytest.raises(ValueError, match="exceeds fixed_length"):
        tok.encode("E300MM00000001")  # 14 文字


def test_encode_rejects_unknown_char() -> None:
    tok = FixedLengthTokenizer()
    with pytest.raises(ValueError, match="not in CHARSET_12H"):
        tok.encode("E300X")


def test_encode_batch_shape() -> None:
    tok = FixedLengthTokenizer()
    targets = tok.encode_batch(["E300MM000001", "", "E300MM999999"])
    assert targets.shape == (3, FIXED_LENGTH)
    assert targets.dtype == torch.long
    assert (targets[1] == EMPTY_IDX).all()


def test_decode_all_empty_returns_empty_string() -> None:
    """全位置 ∅ の logits → 空文字、conf は ∅ 確率の平均 ≈ 1.0。"""
    tok = FixedLengthTokenizer()
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), -10.0)
    logits[0, :, EMPTY_IDX] = 10.0
    results = tok.decode_with_conf(logits)
    assert results[0][0] == ""
    assert results[0][1] > 0.99


def test_decode_full_serial() -> None:
    tok = FixedLengthTokenizer()
    target_text = "E300MM000001"
    target_ids = tok.encode(target_text)
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), -10.0)
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 10.0
    text, conf = tok.decode_with_conf(logits)[0]
    assert text == target_text
    assert conf > 0.99


def test_decode_partial_with_empties() -> None:
    """途中に ∅ を含む → ∅ をスキップして残りを連結。"""
    tok = FixedLengthTokenizer()
    target_ids = [10, 3, 0, 0, EMPTY_IDX, EMPTY_IDX, 0, 0, 0, 0, 0, 1]
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), -10.0)
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 10.0
    text, _ = tok.decode_with_conf(logits)[0]
    assert text == "E300000001"


def test_decode_consistency_with_with_conf() -> None:
    tok = FixedLengthTokenizer()
    torch.manual_seed(0)
    logits = torch.randn(3, FIXED_LENGTH, NUM_CLASSES_12H)
    text_only = tok.decode(logits)
    text_with_conf = [t for t, _ in tok.decode_with_conf(logits)]
    assert text_only == text_with_conf


# ----- Model -----

def test_fixed_head_model_forward_shape() -> None:
    model = FixedHeadOCR(pretrained=False).eval()
    x = torch.randn(2, 1, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, FIXED_LENGTH, NUM_CLASSES_12H)


def test_fixed_head_model_with_3ch_input() -> None:
    model = FixedHeadOCR(pretrained=False).eval()
    x = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, FIXED_LENGTH, NUM_CLASSES_12H)


def test_fixed_head_model_with_rnn_option() -> None:
    model = FixedHeadOCR(pretrained=False, use_rnn=True, rnn_hidden=32).eval()
    x = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, FIXED_LENGTH, NUM_CLASSES_12H)


def test_fixed_head_loss_finite_on_mixed_batch() -> None:
    """positive + negative (= 全位置 ∅) 混在で CrossEntropy loss が有限。"""
    import torch.nn.functional as F
    model = FixedHeadOCR(pretrained=False).eval()
    tok = FixedLengthTokenizer()
    x = torch.randn(3, 1, INPUT_HEIGHT, INPUT_WIDTH)
    targets = tok.encode_batch(["E300MM000001", "", "E300MM999999"])
    with torch.no_grad():
        logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, NUM_CLASSES_12H),
        targets.reshape(-1),
    )
    assert torch.isfinite(loss)


def test_fixed_head_parameter_count() -> None:
    """no_rnn 版が with_rnn 版より小さい (BiGRU 削除効果の確認)。"""
    fh_no_rnn = FixedHeadOCR(pretrained=False)
    fh_with_rnn = FixedHeadOCR(pretrained=False, use_rnn=True, rnn_hidden=64)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    no_rnn = n_params(fh_no_rnn)
    with_rnn = n_params(fh_with_rnn)
    assert no_rnn < with_rnn
    print(f"FixedHeadOCR no_rnn: {no_rnn:,} params")
    print(f"FixedHeadOCR with_rnn (hidden=64): {with_rnn:,} params")
