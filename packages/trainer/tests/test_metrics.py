"""compute_metrics の per-category 集計を検証。"""

from __future__ import annotations

import math

import pytest

from meiban_ocr_trainer.metrics import compute_metrics
from meiban_ocr_trainer.vendors import ERICSSON


def test_all_positive_perfect() -> None:
    preds = ["E300MM000001", "E900MM000002"]
    gts = ["E300MM000001", "E900MM000002"]
    cats = ["positive", "positive"]
    rep = compute_metrics(preds, gts, cats, pattern=ERICSSON.strict_regex)
    assert rep.cer == 0.0
    assert rep.em == 1.0
    assert rep.acceptance_recall == 1.0
    assert rep.em_among_accepted == 1.0
    assert rep.fpr_pattern is None
    assert rep.rejection_recall is None


def test_all_negative_perfect_rejection() -> None:
    preds = ["", "", ""]
    gts = ["", "", ""]
    cats = ["negative"] * 3
    subkinds = ["background", "other_text", "other_text"]
    rep = compute_metrics(preds, gts, cats, subkinds, pattern=ERICSSON.strict_regex)
    assert rep.n_neg == 3
    assert rep.fpr_pattern == 0.0
    assert rep.fpr_nonempty == 0.0
    assert rep.rejection_recall == 1.0
    assert rep.per_subkind["background"]["fpr_pattern"] == 0.0
    assert rep.per_subkind["other_text"]["n"] == 2


def test_negative_falsely_outputs_pattern() -> None:
    preds = ["E300MM999999", ""]
    gts = ["", ""]
    cats = ["negative", "negative"]
    subkinds = ["other_text", "background"]
    rep = compute_metrics(preds, gts, cats, subkinds, pattern=ERICSSON.strict_regex)
    assert rep.fpr_pattern == 0.5
    assert rep.fpr_nonempty == 0.5
    assert rep.rejection_recall == 0.5
    assert rep.per_subkind["other_text"]["fpr_pattern"] == 1.0
    assert rep.per_subkind["background"]["fpr_pattern"] == 0.0


def test_negative_outputs_non_pattern_text() -> None:
    """negative が「何か出すけど pattern 外」→ FPR_pattern=0 だが FPR_nonempty=1。"""
    preds = ["HELLO", "WARNING"]
    gts = ["", ""]
    cats = ["negative", "negative"]
    rep = compute_metrics(preds, gts, cats, pattern=ERICSSON.strict_regex)
    assert rep.fpr_pattern == 0.0
    assert rep.fpr_nonempty == 1.0
    assert rep.rejection_recall == 1.0


def test_mixed_partial_correctness() -> None:
    preds = [
        "E300MM000001",
        "E300MM000003",  # gt: 0002, 1文字違い
        "E300MM999999",  # gt: 0004, pattern OK だが内容違い
        "GARBAGE",       # gt: 0005, pattern 外
        "",              # neg 正しく rejected
        "E900MM111111",  # neg 誤受容
    ]
    gts = [
        "E300MM000001",
        "E300MM000002",
        "E300MM000004",
        "E300MM000005",
        "",
        "",
    ]
    cats = ["positive"] * 4 + ["negative"] * 2
    rep = compute_metrics(preds, gts, cats, pattern=ERICSSON.strict_regex)

    assert rep.n_pos == 4
    assert rep.n_neg == 2
    assert rep.em == 0.25  # 1/4
    assert rep.acceptance_recall == 0.75  # 3/4 pattern合格
    assert math.isclose(rep.em_among_accepted, 1 / 3, abs_tol=1e-6)
    assert rep.fpr_pattern == 0.5
    assert rep.rejection_recall == 0.5


def test_cer_one_character_error() -> None:
    preds = ["E300MM000001"]
    gts = ["E300MM000002"]
    cats = ["positive"]
    rep = compute_metrics(preds, gts, cats, pattern=ERICSSON.strict_regex)
    assert math.isclose(rep.cer, 1 / 12, abs_tol=1e-6)


def test_no_pattern_disables_acceptance_metrics() -> None:
    preds = ["E300MM999999", ""]
    gts = ["", ""]
    cats = ["negative", "negative"]
    rep = compute_metrics(preds, gts, cats, pattern=None)
    assert rep.fpr_nonempty == 0.5
    assert rep.fpr_pattern is None
    assert rep.rejection_recall is None


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(["a"], ["b", "c"], ["positive"])
