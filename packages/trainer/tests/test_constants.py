from meiban_ocr_trainer.constants import BLANK_IDX, CHARSET, NUM_CLASSES


def test_charset_is_36_chars_unique():
    assert len(CHARSET) == 36
    assert len(set(CHARSET)) == 36


def test_charset_matches_digits_then_uppercase():
    assert CHARSET == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_blank_idx_and_num_classes_consistent():
    assert BLANK_IDX == len(CHARSET) == 36
    assert NUM_CLASSES == BLANK_IDX + 1 == 37
