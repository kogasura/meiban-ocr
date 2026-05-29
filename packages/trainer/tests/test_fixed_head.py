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


# ----- New confidence aggregation (min(geomean, min)) -----

def test_confidence_one_weak_position_dominates() -> None:
    """1 位置だけ確率が低い → min(geomean, min) で全体が低くなる (∅ 位置含む集約)。"""
    tok = FixedLengthTokenizer()
    target_text = "E300MM000001"
    target_ids = tok.encode(target_text)
    # 11 位置は強い (≥ 0.99)、1 位置 (pos 11) だけ弱い (~0.30)
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), -10.0)
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 10.0
    # pos 11 を低 confidence に: top1=0.30, 残りに分散
    logits[0, 11, :] = 0.0
    logits[0, 11, target_ids[11]] = 1.0  # top1 がギリギリ
    text, conf = tok.decode_with_conf(logits)[0]
    assert text == target_text
    # 11 位置の top1 prob は softmax(1.0 vs 12 個の 0) ≈ 0.20
    # 旧 (arithmetic mean): ~0.93 で通過
    # 新 (min over all): 0.20 程度 で 弱い
    assert conf < 0.5, f"weak position should dominate, got {conf}"


def test_confidence_uniform_mediocre_logits_give_uniform_conf() -> None:
    """全位置で同じ中程度の logit → geomean ≈ min ≈ 同じ値 (= 一様な不確かさを反映)。"""
    tok = FixedLengthTokenizer()
    # softmax(2.5) / (softmax(2.5) + 12 * softmax(0)) ≈ 0.50
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), 0.0)
    target_ids = tok.encode("E300MM000001")
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 2.5
    text, conf = tok.decode_with_conf(logits)[0]
    assert text == "E300MM000001"
    # 一様 → geomean=min=同じ値、確信度は中程度
    assert 0.4 < conf < 0.6, f"expected ~0.50, got {conf}"


def test_confidence_uniform_strong_gives_high() -> None:
    """全位置で強い top1 → geomean ≈ min ≈ 1.0、合算で高 conf。"""
    tok = FixedLengthTokenizer()
    target_ids = tok.encode("E300MM000001")
    logits = torch.full((1, FIXED_LENGTH, NUM_CLASSES_12H), -10.0)
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 10.0  # ほぼ 1.0
    _, conf = tok.decode_with_conf(logits)[0]
    assert conf > 0.99, f"all-strong should give ~1.0, got {conf}"


def test_confidence_all_empty_with_low_certainty() -> None:
    """全位置 ∅ 出力だが ∅ 確率も低い (混乱状態) → 低 conf。"""
    tok = FixedLengthTokenizer()
    # softmax がほぼ均等な uncertainty 状態 → top1 (= ∅ かも、確率 ~1/13)
    logits = torch.zeros((1, FIXED_LENGTH, NUM_CLASSES_12H))
    logits[0, :, EMPTY_IDX] = 0.1  # ギリギリ ∅ が最大
    text, conf = tok.decode_with_conf(logits)[0]
    assert text == ""
    # top1 prob ≈ softmax([0..0, 0.1]) for the ∅ index ≈ 0.083
    # → conf も 0.08 程度 (低)
    assert conf < 0.2, f"uncertain ∅ should give low conf, got {conf}"


def test_decode_detailed_returns_per_position() -> None:
    """decode_detailed が全位置の prob/margin を返す。"""
    tok = FixedLengthTokenizer()
    logits = torch.zeros((1, FIXED_LENGTH, NUM_CLASSES_12H))
    target_ids = tok.encode("E300MM000001")
    for t, idx in enumerate(target_ids):
        logits[0, t, idx] = 5.0
    result = tok.decode_detailed(logits)[0]
    assert result["text"] == "E300MM000001"
    assert "geomean" in result and "min_top1" in result and "min_margin" in result
    assert len(result["per_position"]) == FIXED_LENGTH
    for pos in result["per_position"]:
        assert "char" in pos and "top1" in pos and "margin" in pos
        assert pos["top1"] >= pos["top2"]


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
