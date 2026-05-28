"""RecognitionDataset と ctc_collate の v2 schema 対応 + CTC empty target 挙動。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from meiban_ocr_trainer.constants import BLANK_IDX, NUM_CLASSES
from meiban_ocr_trainer.data.dataset import RecognitionDataset, ctc_collate
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def _write_image(path: Path, w: int = 128, h: int = 32, fill: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_labels_v2(path: Path, rows: list[list[str]]) -> None:
    header = "filename\ttext\tsplit\tsource\tconfidence\tcategory\tsubkind\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(r) + "\n")


def _write_labels_v1(path: Path, rows: list[list[str]]) -> None:
    header = "filename\ttext\tsplit\tsource\tconfidence\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(r) + "\n")


@pytest.fixture
def v2_dataset_root(tmp_path: Path) -> Path:
    """positive 2 + negative 1 を含む v2 dataset。"""
    root = tmp_path / "data"
    _write_image(root / "train" / "real" / "img_a_l00.png")
    _write_image(root / "train" / "real" / "img_a_l01.png")
    _write_image(root / "train" / "real_neg" / "img_a_n02.png")
    _write_labels_v2(root / "labels.tsv", [
        ["train/real/img_a_l00.png", "E300MM000013", "train", "img_a", "0.99", "positive", ""],
        ["train/real/img_a_l01.png", "E900MM123456", "train", "img_a", "0.95", "positive", ""],
        ["train/real_neg/img_a_n02.png", "", "train", "img_a", "1.0", "negative", "other_text"],
    ])
    return root


def test_dataset_returns_meta_dict_with_category(v2_dataset_root: Path) -> None:
    ds = RecognitionDataset(v2_dataset_root, "train")
    assert len(ds) == 3
    _, meta0 = ds[0]
    assert meta0["text"] == "E300MM000013"
    assert meta0["category"] == "positive"
    assert meta0["subkind"] == ""
    assert meta0["source"] == "img_a"

    _, meta_neg = ds[2]
    assert meta_neg["text"] == ""
    assert meta_neg["category"] == "negative"
    assert meta_neg["subkind"] == "other_text"


def test_dataset_legacy_v1_tsv_defaults_to_positive(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_image(root / "train" / "real" / "img_x.png")
    _write_labels_v1(root / "labels.tsv", [
        ["train/real/img_x.png", "E300MM000001", "train", "img_x", "1.0"],
    ])
    ds = RecognitionDataset(root, "train")
    _, meta = ds[0]
    assert meta["text"] == "E300MM000001"
    assert meta["category"] == "positive"
    assert meta["subkind"] == ""


def test_ctc_collate_passes_category_and_subkinds(v2_dataset_root: Path) -> None:
    ds = RecognitionDataset(v2_dataset_root, "train")
    tok = CTCTokenizer()
    batch = ctc_collate([ds[i] for i in range(len(ds))], tok)

    assert batch["images"].shape[0] == 3
    assert batch["categories"] == ["positive", "positive", "negative"]
    assert batch["subkinds"] == ["", "", "other_text"]
    assert batch["texts"] == ["E300MM000013", "E900MM123456", ""]
    assert batch["target_lengths"].tolist() == [12, 12, 0]
    assert batch["targets"].numel() == 24


def test_ctc_loss_finite_on_mixed_batch(v2_dataset_root: Path) -> None:
    """positive + negative 混在で CTCLoss が NaN/Inf にならない。"""
    ds = RecognitionDataset(v2_dataset_root, "train")
    tok = CTCTokenizer()
    batch = ctc_collate([ds[i] for i in range(len(ds))], tok)

    B = batch["images"].shape[0]
    T = 32
    logits = torch.randn(B, T, NUM_CLASSES, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    input_lengths = torch.full((B,), T, dtype=torch.long)

    loss = F.ctc_loss(
        log_probs, batch["targets"], input_lengths, batch["target_lengths"],
        blank=BLANK_IDX, zero_infinity=True,
    )
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ctc_loss_finite_on_all_negative_batch() -> None:
    tok = CTCTokenizer()
    targets, target_lengths = tok.encode_batch(["", "", ""])
    assert target_lengths.tolist() == [0, 0, 0]
    assert targets.numel() == 0

    B, T = 3, 32
    log_probs = F.log_softmax(torch.randn(T, B, NUM_CLASSES), dim=-1)
    input_lengths = torch.full((B,), T, dtype=torch.long)
    loss = F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=BLANK_IDX, zero_infinity=True,
    )
    assert torch.isfinite(loss)


# ---------- curriculum / ratio control ----------

def test_neg_ratio_for_epoch_endpoints_and_interpolation() -> None:
    from meiban_ocr_trainer.data.dataset import neg_ratio_for_epoch

    sched = [
        {"epoch": 1, "ratio": 0.0},
        {"epoch": 5, "ratio": 0.0},
        {"epoch": 15, "ratio": 0.30},
        {"epoch": 40, "ratio": 0.40},
    ]
    # 端点・範囲外
    assert neg_ratio_for_epoch(sched, 0) == 0.0
    assert neg_ratio_for_epoch(sched, 1) == 0.0
    assert neg_ratio_for_epoch(sched, 100) == 0.40

    # warmup 範囲は 0.0 維持
    assert neg_ratio_for_epoch(sched, 3) == 0.0
    assert neg_ratio_for_epoch(sched, 5) == 0.0

    # 5→15 で 0 → 0.30 の線形補間
    assert abs(neg_ratio_for_epoch(sched, 10) - 0.15) < 1e-6
    assert abs(neg_ratio_for_epoch(sched, 15) - 0.30) < 1e-6

    # 15→40 で 0.30 → 0.40 の線形補間
    assert abs(neg_ratio_for_epoch(sched, 25) - (0.30 + (10 / 25) * 0.10)) < 1e-6


def test_neg_ratio_empty_schedule_returns_zero() -> None:
    from meiban_ocr_trainer.data.dataset import neg_ratio_for_epoch
    assert neg_ratio_for_epoch([], 5) == 0.0


def test_curriculum_sampler_respects_neg_ratio(v2_dataset_root: Path) -> None:
    """build_train_loader_with_ratio で neg_ratio=0.5 を強制したとき、サンプリングが
    数値的にだいたい半々になる (n_samples を大きめに取って統計的に確認)。
    """
    from meiban_ocr_trainer.data.dataset import (
        RecognitionDataset,
        build_train_loader_with_ratio,
    )

    tok = CTCTokenizer()
    ds = RecognitionDataset(v2_dataset_root, "train", tok)
    # 統計安定化のため batch_size=10、num_samples=1000 で 100 ステップ回す
    loader = build_train_loader_with_ratio(
        ds, tok, batch_size=10, neg_ratio=0.5,
        num_workers=0, num_samples=1000,
    )

    cat_counter: dict[str, int] = {"positive": 0, "negative": 0}
    for batch in loader:
        for c in batch["categories"]:
            cat_counter[c] += 1
    total = sum(cat_counter.values())
    neg_frac = cat_counter["negative"] / total
    # 0.5 ± 0.05 程度を許容 (n=1000 のばらつき)
    assert 0.45 < neg_frac < 0.55, f"neg_frac={neg_frac:.3f}"


def test_curriculum_sampler_zero_ratio_yields_only_positive(v2_dataset_root: Path) -> None:
    """neg_ratio=0.0 のとき (warmup 期) negative は全く混入しない。"""
    from meiban_ocr_trainer.data.dataset import (
        RecognitionDataset,
        build_train_loader_with_ratio,
    )

    tok = CTCTokenizer()
    ds = RecognitionDataset(v2_dataset_root, "train", tok)
    loader = build_train_loader_with_ratio(
        ds, tok, batch_size=10, neg_ratio=0.0,
        num_workers=0, num_samples=200,
    )
    n_neg = 0
    for batch in loader:
        n_neg += sum(1 for c in batch["categories"] if c == "negative")
    assert n_neg == 0
