import torch

from meiban_ocr_trainer.constants import BLANK_IDX, NUM_CLASSES
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def test_encode_known_serial():
    tok = CTCTokenizer()
    # 12文字シリアル (dummy E300MM000032、real serial は使わない)
    ids = tok.encode("E300MM000032")
    # E=14, 3=3, 0=0, 0=0, M=22, M=22, 0=0, 0=0, 0=0, 0=0, 3=3, 2=2
    assert ids == [14, 3, 0, 0, 22, 22, 0, 0, 0, 0, 3, 2]


def test_encode_rejects_unknown_char():
    tok = CTCTokenizer()
    try:
        tok.encode("hello-world")
    except ValueError:
        return
    raise AssertionError("expected ValueError for lowercase input")


def test_encode_batch_lengths():
    tok = CTCTokenizer()
    flat, lengths = tok.encode_batch(["AB", "Z9X"])
    assert lengths.tolist() == [2, 3]
    assert flat.numel() == 5


def test_greedy_decode_collapses_blanks_and_repeats():
    tok = CTCTokenizer()
    # B=1, T=8, C=37。シーケンス: A, A, blank, B, B, blank, blank, C  → "ABC"
    A = 10  # 'A'
    B = 11
    C = 12
    seq = [A, A, BLANK_IDX, B, B, BLANK_IDX, BLANK_IDX, C]
    logits = torch.full((1, len(seq), NUM_CLASSES), -10.0)
    for t, idx in enumerate(seq):
        logits[0, t, idx] = 10.0
    out = tok.greedy_decode(logits)
    assert out == ["ABC"]


def test_greedy_decode_with_conf_high_for_strong_logits():
    """非常に強い logit (softmax ≈ 1) のとき confidence は ≈ 1.0。"""
    tok = CTCTokenizer()
    A = 10
    seq = [A, A, BLANK_IDX, BLANK_IDX]
    logits = torch.full((1, len(seq), NUM_CLASSES), -10.0)
    for t, idx in enumerate(seq):
        logits[0, t, idx] = 10.0
    results = tok.greedy_decode_with_conf(logits)
    assert len(results) == 1
    text, conf = results[0]
    assert text == "A"
    assert conf > 0.99  # softmax(10) over (10, -10, ...) → ~1.0


def test_greedy_decode_with_conf_low_for_uncertain_logits():
    """logit が均等 (flat) のとき confidence は低い (≈ 1/NUM_CLASSES)。"""
    tok = CTCTokenizer()
    # 全 timestep で blank が辛うじて max だが logit はほぼ均等
    logits = torch.zeros((1, 4, NUM_CLASSES))
    logits[0, :, BLANK_IDX] = 0.1  # わずかに blank が最大
    results = tok.greedy_decode_with_conf(logits)
    text, conf = results[0]
    assert text == ""  # 全 blank
    # 空出力の confidence は blank prob 平均 → softmax で計算すると小さい
    assert conf < 0.1


def test_greedy_decode_with_conf_empty_returns_reject_confidence():
    """全 blank なら text="" + blank softmax の平均 confidence。"""
    tok = CTCTokenizer()
    logits = torch.full((1, 4, NUM_CLASSES), -10.0)
    logits[0, :, BLANK_IDX] = 10.0  # 全 timestep が blank
    results = tok.greedy_decode_with_conf(logits)
    text, conf = results[0]
    assert text == ""
    assert conf > 0.99  # blank prob ≈ 1


def test_greedy_decode_consistency_with_with_conf():
    """greedy_decode と greedy_decode_with_conf のテキストが一致する。"""
    tok = CTCTokenizer()
    torch.manual_seed(0)
    logits = torch.randn(3, 16, NUM_CLASSES)
    texts_only = tok.greedy_decode(logits)
    texts_with_conf = [t for t, _ in tok.greedy_decode_with_conf(logits)]
    assert texts_only == texts_with_conf
