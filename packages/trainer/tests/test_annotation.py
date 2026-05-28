"""annotation.py の v1/v2 schema 互換と round-trip を検証。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meiban_ocr_trainer.data.annotation import (
    SCHEMA_VERSION,
    Annotation,
    Region,
    load_annotation,
    save_annotation,
)


def test_load_v1_labels_converted_to_positive_regions(tmp_path: Path) -> None:
    raw = {
        "image": "img_001.jpg",
        "image_size": [1280, 960],
        "source_video": None,
        "vendor": "ericsson",
        "labels": [
            {
                "id": 0, "bbox": [10, 20, 100, 50],
                "text_bbox": [12, 22, 98, 48], "text": "E300MM000013",
                "confidence": 0.99, "is_clear": True,
                "match_kind": "strict", "claude_verified": True,
            },
            {
                "id": 1, "bbox": [200, 200, 300, 230],
                "text_bbox": [202, 202, 298, 228], "text": "E300MM000014",
                "confidence": 0.72, "is_clear": False,
                "match_kind": "strict_O_to_0", "claude_verified": False,
            },
        ],
        "ocr_engine": "rapidocr_onnxruntime",
        "rejected_by_regex_primary": 5,
    }
    p = tmp_path / "img_001.json"
    p.write_text(json.dumps(raw))

    ann = load_annotation(p)
    assert ann.image == "img_001.jpg"
    assert ann.vendor == "ericsson"
    assert ann.schema_version == 1
    assert len(ann.regions) == 2
    assert all(r.category == "positive" for r in ann.regions)
    assert ann.regions[0].quality == "clear"
    assert ann.regions[1].quality == "blur"
    assert ann.regions[0].vendor == "ericsson"
    assert ann.meta["ocr_engine"] == "rapidocr_onnxruntime"
    assert ann.meta["rejected_by_regex_primary"] == 5


def test_load_v2_regions_positive_and_negative(tmp_path: Path) -> None:
    raw = {
        "image": "img_002.jpg",
        "image_size": [1024, 768],
        "vendor": "ericsson",
        "schema_version": 2,
        "regions": [
            {
                "id": 0, "category": "positive",
                "bbox": [10, 10, 100, 40], "text_bbox": [12, 12, 98, 38],
                "text": "E300MM999003", "quality": "clear",
                "confidence": 0.95, "claude_verified": True,
            },
            {
                "id": 1, "category": "negative",
                "bbox": [200, 300, 350, 340],
                "subkind": "other_text", "text": "",
                "text_visible": "WARNING", "claude_verified": True,
            },
            {
                "id": 2, "category": "negative",
                "bbox": [500, 500, 632, 540],
                "subkind": "background", "text": "",
            },
        ],
    }
    p = tmp_path / "img_002.json"
    p.write_text(json.dumps(raw))

    ann = load_annotation(p)
    assert len(ann.regions) == 3
    assert len(ann.positives) == 1
    assert len(ann.negatives) == 2
    assert ann.positives[0].text == "E300MM999003"
    assert ann.negatives[0].subkind == "other_text"
    assert ann.negatives[0].text_visible == "WARNING"
    assert ann.negatives[1].subkind == "background"
    assert ann.schema_version == 2


def test_save_writes_v2_format(tmp_path: Path) -> None:
    ann = Annotation(
        image="img_003.jpg",
        image_size=[800, 600],
        vendor="ericsson",
        regions=[
            Region(
                id=0, category="positive", bbox=[10, 10, 100, 40],
                text_bbox=[12, 12, 98, 38], text="E300MM000001",
                quality="clear", confidence=0.99, claude_verified=True,
            ),
            Region(
                id=1, category="negative", bbox=[200, 200, 300, 240],
                subkind="background", claude_verified=True,
            ),
        ],
    )
    p = tmp_path / "out.json"
    save_annotation(ann, p)

    data = json.loads(p.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert "regions" in data and "labels" not in data
    assert len(data["regions"]) == 2
    assert data["regions"][0]["category"] == "positive"
    assert data["regions"][0]["text"] == "E300MM000001"
    assert data["regions"][1]["category"] == "negative"
    assert data["regions"][1]["text"] == ""
    assert data["regions"][1]["subkind"] == "background"


def test_round_trip_v1_to_v2_preserves_data(tmp_path: Path) -> None:
    v1 = {
        "image": "img.jpg",
        "image_size": [1280, 960],
        "vendor": "ericsson",
        "labels": [
            {
                "id": 0, "bbox": [10, 10, 100, 40],
                "text_bbox": [12, 12, 98, 38], "text": "E300MM000013",
                "confidence": 0.99, "is_clear": True,
                "match_kind": "strict", "claude_verified": True,
            },
        ],
        "ocr_engine": "rapidocr_onnxruntime",
        "claude_verification": {"verifier": "claude-opus-4-7", "result": "all_correct"},
    }
    p_in = tmp_path / "in.json"
    p_in.write_text(json.dumps(v1))

    ann = load_annotation(p_in)
    p_out = tmp_path / "out.json"
    save_annotation(ann, p_out)

    ann2 = load_annotation(p_out)
    assert ann2.schema_version == SCHEMA_VERSION
    assert len(ann2.positives) == 1
    r = ann2.positives[0]
    assert r.text == "E300MM000013"
    assert r.bbox == [10, 10, 100, 40]
    assert r.text_bbox == [12, 12, 98, 38]
    assert r.quality == "clear"
    assert r.match_kind == "strict"
    assert r.confidence == 0.99
    assert r.claude_verified is True
    assert ann2.meta["ocr_engine"] == "rapidocr_onnxruntime"
    assert ann2.meta["claude_verification"]["result"] == "all_correct"


def test_negative_with_text_raises() -> None:
    with pytest.raises(ValueError, match="must have empty text"):
        Region(id=0, category="negative", bbox=[0, 0, 10, 10], text="OOPS")


def test_positive_without_text_raises() -> None:
    with pytest.raises(ValueError, match="must have non-empty text"):
        Region(id=0, category="positive", bbox=[0, 0, 10, 10], text="")


def test_load_missing_regions_and_labels_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"image": "x.jpg", "image_size": [1, 1]}))
    with pytest.raises(ValueError, match="neither regions"):
        load_annotation(p)


def test_load_missing_image_size_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"image": "x.jpg", "labels": []}))
    with pytest.raises(ValueError, match="missing image"):
        load_annotation(p)


def test_text_visible_rejects_real_serial_pattern() -> None:
    """text_visible に Ericsson strict pattern を含むリアルシリアルを書くと ValueError。"""
    with pytest.raises(ValueError, match="forbidden pattern"):
        Region(
            id=0, category="negative", bbox=[0, 0, 10, 10],
            subkind="other_text",
            # E300MM* (dummy 範囲) でも strict regex に一致するので validator は弾く
            text_visible="E300MM999003",
        )


def test_text_visible_rejects_long_digit_sequence() -> None:
    """5 桁以上の連続数字も弾く (シリアル末尾の可能性)。"""
    with pytest.raises(ValueError, match="forbidden pattern"):
        Region(
            id=1, category="negative", bbox=[0, 0, 10, 10],
            subkind="other_text",
            text_visible="503509 (handwritten)",
        )


def test_text_visible_allows_safe_strings() -> None:
    """公開情報や redact 済表記は通る。"""
    # 製品モデル名 / 法人名
    Region(
        id=2, category="negative", bbox=[0, 0, 10, 10],
        subkind="other_text",
        text_visible="Radio 2218 B42B",
    )
    Region(
        id=3, category="negative", bbox=[0, 0, 10, 10],
        subkind="other_text",
        text_visible="エリクソン・ジャパン株式会社",
    )
    # redact 済
    Region(
        id=4, category="negative", bbox=[0, 0, 10, 10],
        subkind="other_text",
        text_visible="(handwritten digits, redacted)",
    )
    # 4 桁までは許容 (年号・短い ID 等)
    Region(
        id=5, category="negative", bbox=[0, 0, 10, 10],
        subkind="other_text",
        text_visible="2018年9月",
    )


def test_text_visible_none_skips_validation() -> None:
    """text_visible 未指定なら validator は走らない (positive region 等)。"""
    Region(
        id=6, category="positive", bbox=[0, 0, 10, 10],
        text="E300MM000001",
    )


def test_unknown_region_fields_preserved_in_extra(tmp_path: Path) -> None:
    raw = {
        "image": "x.jpg",
        "image_size": [10, 10],
        "schema_version": 2,
        "regions": [
            {
                "id": 0, "category": "positive",
                "bbox": [0, 0, 5, 5], "text": "A",
                "mined_from": "run_2026_05_28",
                "iou_with_secondary": 0.92,
            }
        ],
    }
    p = tmp_path / "x.json"
    p.write_text(json.dumps(raw))
    ann = load_annotation(p)
    r = ann.regions[0]
    assert r.extra["mined_from"] == "run_2026_05_28"
    assert r.extra["iou_with_secondary"] == 0.92

    p2 = tmp_path / "x_out.json"
    save_annotation(ann, p2)
    raw2 = json.loads(p2.read_text())
    assert raw2["regions"][0]["mined_from"] == "run_2026_05_28"
