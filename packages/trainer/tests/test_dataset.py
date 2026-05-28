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
