import torch

from meiban_ocr_trainer.constants import BLANK_IDX, NUM_CLASSES
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def test_encode_known_serial():
    tok = CTCTokenizer()
    ids = tok.encode("E300MM000032")
    # E=14, 3=3, 0=0, 3=3, M=22, M=22, 5=5, 0=0, 0=0, 9=9, 4=4, 2=2
    assert ids == [14, 3, 0, 3, 22, 22, 5, 0, 0, 9, 4, 2]


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
