"""Annotation schema v2: `regions[]` (positive + negative 統合) と loader/saver。

旧 v1 schema (`labels[]` only) も `load_annotation()` 経由で読める (category=positive
として変換)。新規ファイルは常に v2 形式で書き出す。

v2 schema:
    {
      "image": "img_001.jpg",
      "image_size": [W, H],
      "source_video": null | str,
      "vendor": "ericsson" | null,        # 画像全体のデフォルトベンダー
      "schema_version": 2,
      "regions": [
        # positive: 認識器の学習対象
        {
          "id": 0,
          "category": "positive",
          "bbox": [x1, y1, x2, y2],          # crop に使うパディング込み bbox
          "text_bbox": [x1, y1, x2, y2],     # tight な text 境界 (optional)
          "text": "E300MM000013",
          "vendor": "ericsson",              # optional: 画像 vendor の override
          "quality": "clear",                # "clear" | "blur" | "partial" | "occluded"
          "confidence": 0.99,                # ラベリング時の信頼度
          "match_kind": "strict",
          "claude_verified": true
        },
        # negative: reject 訓練対象。text は常に空。
        {
          "id": 1,
          "category": "negative",
          "bbox": [x1, y1, x2, y2],
          "subkind": "other_text",           # background | other_text | partial | other_vendor | mined
          "text_visible": "WARNING",         # 参考メモ。学習には使わない
          "claude_verified": true
        }
      ],
      # 任意メタ (auto_label が付与する OCR 情報など) は meta に温存
      "ocr_engine": "...", "ocr_elapsed_sec": {...},
      "claude_verification": {...}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Category = Literal["positive", "negative"]
NegativeSubkind = Literal["background", "other_text", "partial", "other_vendor", "mined"]
Quality = Literal["clear", "blur", "partial", "occluded"]

SCHEMA_VERSION = 2

# 画像レベルのメタキー (loader/saver で破壊しない)
_PRESERVED_META_KEYS = (
    "ocr_engine",
    "mode",
    "ocr_elapsed_sec",
    "rejected_by_regex_primary",
    "claude_verification",
    "secondary_engine",
    "secondary_candidates_count",
    "disagreements",
)


@dataclass
class Region:
    """画像内の 1 領域。positive は認識学習対象、negative は reject 訓練対象。"""

    id: int
    category: Category
    bbox: list[int]  # [x1, y1, x2, y2], crop に使う範囲 (negative も同様)
    # positive 専用フィールド
    text: str = ""  # negative は常に ""
    text_bbox: list[int] | None = None
    vendor: str | None = None
    quality: Quality | None = None
    confidence: float | None = None
    match_kind: str | None = None
    # negative 専用フィールド
    subkind: NegativeSubkind | None = None
    text_visible: str | None = None  # 参考メモ。学習データには出さない
    # 共通
    claude_verified: bool = False
    # 拡張用 (将来フィールド・mining 由来情報など)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category == "negative" and self.text:
            raise ValueError(
                f"negative region id={self.id} must have empty text, got {self.text!r}"
            )
        if self.category == "positive" and not self.text:
            raise ValueError(
                f"positive region id={self.id} must have non-empty text"
            )


@dataclass
class Annotation:
    """画像 1枚分のアノテーション。"""

    image: str
    image_size: list[int]  # [W, H]
    source_video: str | None = None
    vendor: str | None = None
    regions: list[Region] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    # 画像レベルメタの温存 (auto_label が付ける OCR 計測値など)
    meta: dict = field(default_factory=dict)

    @property
    def positives(self) -> list[Region]:
        return [r for r in self.regions if r.category == "positive"]

    @property
    def negatives(self) -> list[Region]:
        return [r for r in self.regions if r.category == "negative"]


def _region_from_v2(d: dict) -> Region:
    return Region(
        id=int(d["id"]),
        category=d["category"],
        bbox=list(d["bbox"]),
        text=d.get("text", ""),
        text_bbox=list(d["text_bbox"]) if d.get("text_bbox") is not None else None,
        vendor=d.get("vendor"),
        quality=d.get("quality"),
        confidence=d.get("confidence"),
        match_kind=d.get("match_kind"),
        subkind=d.get("subkind"),
        text_visible=d.get("text_visible"),
        claude_verified=bool(d.get("claude_verified", False)),
        extra={
            k: v for k, v in d.items()
            if k not in {
                "id", "category", "bbox", "text", "text_bbox", "vendor",
                "quality", "confidence", "match_kind", "subkind",
                "text_visible", "claude_verified",
            }
        },
    )


def _region_from_v1_label(d: dict, default_vendor: str | None) -> Region:
    """旧 labels[] エントリを positive Region に変換。

    旧 schema は positive のみ。is_clear: True/False を quality: "clear"/"blur" に対応付け。
    """
    is_clear = d.get("is_clear", True)
    quality: Quality = "clear" if is_clear else "blur"
    return Region(
        id=int(d["id"]),
        category="positive",
        bbox=list(d["bbox"]),
        text=d["text"],
        text_bbox=list(d["text_bbox"]) if d.get("text_bbox") is not None else None,
        vendor=default_vendor,
        quality=quality,
        confidence=d.get("confidence"),
        match_kind=d.get("match_kind"),
        claude_verified=bool(d.get("claude_verified", False)),
        extra={
            k: v for k, v in d.items()
            if k not in {
                "id", "bbox", "text", "text_bbox", "confidence", "is_clear",
                "match_kind", "claude_verified",
            }
        },
    )


def load_annotation(path: Path) -> Annotation:
    """JSON を読んで Annotation を返す。v1/v2 両対応。

    判定優先順位:
      1. `regions` キーがあれば v2
      2. `labels` キーがあれば v1 (全部 positive に変換)
      3. どちらも無ければエラー
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"annotation root must be object: {path}")

    image = data.get("image")
    image_size = data.get("image_size")
    if image is None or image_size is None:
        raise ValueError(f"annotation missing image/image_size: {path}")

    vendor = data.get("vendor")
    regions: list[Region] = []

    if "regions" in data:
        for r in data["regions"]:
            regions.append(_region_from_v2(r))
    elif "labels" in data:
        for lbl in data["labels"]:
            regions.append(_region_from_v1_label(lbl, default_vendor=vendor))
    else:
        raise ValueError(
            f"annotation has neither regions[] nor labels[]: {path}"
        )

    meta = {k: data[k] for k in _PRESERVED_META_KEYS if k in data}

    return Annotation(
        image=image,
        image_size=list(image_size),
        source_video=data.get("source_video"),
        vendor=vendor,
        regions=regions,
        schema_version=int(
            data.get("schema_version", 1 if "labels" in data else SCHEMA_VERSION)
        ),
        meta=meta,
    )


def _region_to_dict(r: Region) -> dict:
    """Region を JSON 書き出し用 dict に変換。None フィールドは省略。"""
    out: dict = {
        "id": r.id,
        "category": r.category,
        "bbox": list(r.bbox),
    }
    if r.text_bbox is not None:
        out["text_bbox"] = list(r.text_bbox)
    # negative も text="" を明示的に書く
    out["text"] = r.text if r.category == "positive" else ""
    if r.subkind is not None:
        out["subkind"] = r.subkind
    if r.text_visible is not None:
        out["text_visible"] = r.text_visible
    if r.vendor is not None:
        out["vendor"] = r.vendor
    if r.quality is not None:
        out["quality"] = r.quality
    if r.confidence is not None:
        out["confidence"] = r.confidence
    if r.match_kind is not None:
        out["match_kind"] = r.match_kind
    if r.claude_verified:
        out["claude_verified"] = True
    for k, v in r.extra.items():
        if k not in out:
            out[k] = v
    return out


def save_annotation(ann: Annotation, path: Path) -> None:
    """Annotation を常に v2 形式で書き出す。

    画像レベルメタ (`meta` フィールド) はキー順を保ちつつ末尾に追加。
    """
    out: dict = {
        "image": ann.image,
        "image_size": list(ann.image_size),
        "source_video": ann.source_video,
        "vendor": ann.vendor,
        "schema_version": SCHEMA_VERSION,
        "regions": [_region_to_dict(r) for r in ann.regions],
    }
    for k in _PRESERVED_META_KEYS:
        if k in ann.meta:
            out[k] = ann.meta[k]
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Annotation",
    "Region",
    "Category",
    "NegativeSubkind",
    "Quality",
    "SCHEMA_VERSION",
    "load_annotation",
    "save_annotation",
]
