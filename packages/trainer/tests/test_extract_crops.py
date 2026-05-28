"""extract_crops.py の v2 schema 対応 + negative 出力を検証。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meiban_ocr_trainer.data.annotation import (
    Annotation,
    Region,
    save_annotation,
)
from meiban_ocr_trainer.data.extract_crops import (
    LABELS_TSV_HEADER,
    NEGATIVE_SUBDIR,
    POSITIVE_SUBDIR,
    extract_crops,
)


def _make_synthetic_image(path: Path, w: int = 200, h: int = 100) -> None:
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[20:50, 20:80] = (40, 40, 40)
    arr[60:90, 100:180] = (80, 80, 80)
    Image.fromarray(arr).save(path)


def _make_annotation(image_name: str, w: int, h: int) -> Annotation:
    """positive 1件 + negative 2件 (other_text / background) を含む。"""
    return Annotation(
        image=image_name,
        image_size=[w, h],
        vendor="ericsson",
        regions=[
            Region(
                id=0, category="positive",
                bbox=[15, 15, 85, 55], text_bbox=[20, 20, 80, 50],
                text="E300MM000013", quality="clear", confidence=0.99,
                match_kind="strict", claude_verified=True,
            ),
            Region(
                id=1, category="negative",
                bbox=[100, 60, 180, 90], subkind="other_text",
                text_visible="WARNING", claude_verified=True,
            ),
            Region(
                id=2, category="negative",
                bbox=[5, 5, 15, 15], subkind="background",
                claude_verified=True,
            ),
        ],
    )


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Path]:
    samples = tmp_path / "samples"
    annotations = tmp_path / "annotations"
    out = tmp_path / "data" / "recognition"
    samples.mkdir()
    annotations.mkdir()

    img_path = samples / "img_001.jpg"
    _make_synthetic_image(img_path, w=200, h=100)

    ann = _make_annotation("img_001.jpg", 200, 100)
    save_annotation(ann, annotations / "img_001.json")

    return {"samples": samples, "annotations": annotations, "out": out}


def test_extract_writes_positive_and_negative_crops(repo: dict[str, Path]) -> None:
    split_map = {"train": {"img_001"}, "val": set(), "test": set()}
    counts = extract_crops(
        repo["samples"], repo["annotations"], repo["out"],
        split_map=split_map, require_verified=True,
    )

    assert counts["_total"] == 3
    assert counts["_pos_total"] == 1
    assert counts["_neg_total"] == 2
    assert counts["train_pos"] == 1
    assert counts["train_neg"] == 2

    pos_dir = repo["out"] / "train" / POSITIVE_SUBDIR
    neg_dir = repo["out"] / "train" / NEGATIVE_SUBDIR
    assert (pos_dir / "img_001_l00.png").exists()
    assert (neg_dir / "img_001_n01.png").exists()
    assert (neg_dir / "img_001_n02.png").exists()


def test_labels_tsv_has_v2_columns_and_correct_rows(repo: dict[str, Path]) -> None:
    split_map = {"train": {"img_001"}, "val": set(), "test": set()}
    extract_crops(
        repo["samples"], repo["annotations"], repo["out"],
        split_map=split_map, require_verified=True,
    )

    import csv
    with (repo["out"] / "labels.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames == LABELS_TSV_HEADER
        rows = list(reader)

    assert len(rows) == 3
    pos = [r for r in rows if r["category"] == "positive"]
    neg = [r for r in rows if r["category"] == "negative"]
    assert len(pos) == 1 and len(neg) == 2

    assert pos[0]["text"] == "E300MM000013"
    assert pos[0]["subkind"] == ""

    assert all(r["text"] == "" for r in neg)
    subkinds = sorted(r["subkind"] for r in neg)
    assert subkinds == ["background", "other_text"]
    assert all(r["filename"].split("/")[1] == NEGATIVE_SUBDIR for r in neg)


def test_require_verified_filters_unverified_regions(repo: dict[str, Path]) -> None:
    ann = _make_annotation("img_001.jpg", 200, 100)
    ann.regions[1].claude_verified = False
    save_annotation(ann, repo["annotations"] / "img_001.json")

    split_map = {"train": {"img_001"}, "val": set(), "test": set()}
    counts = extract_crops(
        repo["samples"], repo["annotations"], repo["out"],
        split_map=split_map, require_verified=True,
    )
    assert counts["_total"] == 2
    assert counts["_neg_total"] == 1


def test_unverified_included_when_flag_off(repo: dict[str, Path]) -> None:
    ann = _make_annotation("img_001.jpg", 200, 100)
    ann.regions[1].claude_verified = False
    save_annotation(ann, repo["annotations"] / "img_001.json")

    split_map = {"train": {"img_001"}, "val": set(), "test": set()}
    counts = extract_crops(
        repo["samples"], repo["annotations"], repo["out"],
        split_map=split_map, require_verified=False,
    )
    assert counts["_total"] == 3


def test_split_assignment(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    annotations = tmp_path / "annotations"
    out = tmp_path / "out"
    samples.mkdir()
    annotations.mkdir()

    for stem in ("img_a", "img_b"):
        _make_synthetic_image(samples / f"{stem}.jpg")
        save_annotation(
            _make_annotation(f"{stem}.jpg", 200, 100),
            annotations / f"{stem}.json",
        )

    split_map = {"train": {"img_a"}, "val": {"img_b"}, "test": set()}
    counts = extract_crops(samples, annotations, out, split_map=split_map)
    assert counts["train"] == 3
    assert counts["val"] == 3
    assert counts["test"] == 0
    assert (out / "train" / POSITIVE_SUBDIR / "img_a_l00.png").exists()
    assert (out / "val" / POSITIVE_SUBDIR / "img_b_l00.png").exists()


def test_v1_annotation_still_works(tmp_path: Path) -> None:
    """旧 labels[] schema の annotation も load 経由で読めて crop できる。"""
    import json
    samples = tmp_path / "samples"
    annotations = tmp_path / "annotations"
    out = tmp_path / "out"
    samples.mkdir()
    annotations.mkdir()

    _make_synthetic_image(samples / "img_old.jpg", w=200, h=100)
    v1 = {
        "image": "img_old.jpg",
        "image_size": [200, 100],
        "vendor": "ericsson",
        "labels": [
            {
                "id": 0, "bbox": [15, 15, 85, 55],
                "text_bbox": [20, 20, 80, 50], "text": "E300MM000013",
                "confidence": 0.99, "is_clear": True,
                "match_kind": "strict", "claude_verified": True,
            },
        ],
    }
    (annotations / "img_old.json").write_text(json.dumps(v1))

    counts = extract_crops(
        samples, annotations, out,
        split_map={"train": {"img_old"}, "val": set(), "test": set()},
    )
    assert counts["_pos_total"] == 1
    assert counts["_neg_total"] == 0
