"""Stage 1 (samples/ + annotations/*.json) → Stage 2 (data/recognition/{train,val,test}) 変換。

各 region (positive/negative) を切り出し、PNG + labels.tsv を生成。
- positive: text_bbox を少しパディングして crop → `{split}/real/`
- negative: bbox をそのまま crop → `{split}/real_neg/`

annotation は v2 schema (`regions[]`) を前提。v1 (`labels[]`) も loader 経由で読める。

train/val/test 分割は **画像単位** で行う (HANDOFF.md §4 Step 1 注意点: 同じ画像から
両方に振らない)。

v0 (4枚 / 54ラベル) のデフォルト分割:
- train: img_001 (RRUS 11 B1) + img_003 (RRU 22F3) → 38 labels
- val:   img_002 (Radio 2218 B42B) → 13 labels
- test:  img_004 (Radio 2251 B18 B280) → 3 labels

Usage:
    python -m meiban_ocr_trainer.data.extract_crops \\
        --samples-dir samples \\
        --annotations-dir annotations \\
        --output-dir data/recognition

labels.tsv schema:
    filename  text  split  source  confidence  category  subkind
    - category: positive | negative  (旧 tsv で欠落していたら positive とみなす)
    - subkind:  negative の場合のみ (background | other_text | partial | other_vendor | mined)
    - text:     negative は常に空文字
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from meiban_ocr_trainer.data.annotation import Annotation, Region, load_annotation

# v0 デフォルト分割 (画像単位)
DEFAULT_SPLIT: dict[str, set[str]] = {
    "train": {"img_001", "img_003"},
    "val": {"img_002"},
    "test": {"img_004"},
}

# サブディレクトリ名 (positive/negative で分ける)
POSITIVE_SUBDIR = "real"
NEGATIVE_SUBDIR = "real_neg"

# labels.tsv のカラム順
LABELS_TSV_HEADER = [
    "filename", "text", "split", "source", "confidence", "category", "subkind",
]


@dataclass
class CropRecord:
    split: str
    source_image: str
    region_id: int
    text: str  # negative は ""
    crop_relpath: str  # labels.tsv に書く相対パス
    confidence: float  # negative は 1.0 (信頼度の概念がないため固定)
    category: str  # "positive" | "negative"
    subkind: str  # negative の subkind、positive は ""


def _resolve_split(image_stem: str, split_map: dict[str, set[str]]) -> str | None:
    for split, stems in split_map.items():
        if image_stem in stems:
            return split
    return None


def crop_positive(
    image: Image.Image,
    region: Region,
    margin_px: int = 4,
) -> Image.Image:
    """positive: text_bbox を少しだけ padding してクロップ (CRNN 1行テキスト認識用)。

    Why text_bbox instead of bbox: CRNN+CTC は単一行テキスト認識器であり、複数行が写った
    クロップを学習させると CTC の整合がとれない (ground truth は serial 12文字なのに画像
    には RRU 22F3 / シリアル / 日付 / 会社名 の4行が見える)。
    text_bbox は serial だけを囲んでおり、CRNN のトレーニング・推論の両方に整合する。
    text_bbox が無ければ bbox にフォールバック。
    """
    bbox = region.text_bbox if region.text_bbox is not None else region.bbox
    x1, y1, x2, y2 = bbox
    w, h = image.size
    return image.crop((
        max(0, x1 - margin_px),
        max(0, y1 - margin_px),
        min(w, x2 + margin_px),
        min(h, y2 + margin_px),
    ))


def crop_negative(image: Image.Image, region: Region) -> Image.Image:
    """negative: bbox をそのまま crop。

    Why no margin: negative の bbox は「ここはコードではない」領域として既にアノテーター
    が定義しており、パディングで positive 領域を巻き込むリスクがある。
    """
    x1, y1, x2, y2 = region.bbox
    w, h = image.size
    return image.crop((
        max(0, x1), max(0, y1), min(w, x2), min(h, y2),
    ))


def extract_crops(
    samples_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    split_map: dict[str, set[str]] | None = None,
    require_verified: bool = True,
) -> dict[str, int]:
    """Stage 1 → Stage 2 変換のメインルーチン。

    Args:
        samples_dir: 元画像ディレクトリ
        annotations_dir: annotations/*.json
        output_dir: data/recognition (この下に train/val/test を切る)
        split_map: image_stem → split の対応。未指定なら DEFAULT_SPLIT
        require_verified: True の場合、claude_verified=False の region をスキップ
                          (positive/negative 両方に適用)

    Returns:
        各 split のサンプル数辞書 (`_total`, `_pos_total`, `_neg_total` 含む)
    """
    split_map = split_map or DEFAULT_SPLIT
    for split in split_map:
        (output_dir / split / POSITIVE_SUBDIR).mkdir(parents=True, exist_ok=True)
        # negative ディレクトリは出現時のみ作る (空ディレクトリを残さない)

    records: list[CropRecord] = []

    for ann_path in sorted(annotations_dir.glob("img_*.json")):
        ann: Annotation = load_annotation(ann_path)
        image_stem = Path(ann.image).stem
        split = _resolve_split(image_stem, split_map)
        if split is None:
            print(f"  - {image_stem}: no split assigned, skip", file=sys.stderr)
            continue

        image_path = samples_dir / ann.image
        if not image_path.exists():
            print(f"  - {image_stem}: missing image {image_path}", file=sys.stderr)
            continue

        image = Image.open(image_path).convert("RGB")

        n_pos = 0
        n_neg = 0
        for region in ann.regions:
            if require_verified and not region.claude_verified:
                continue

            if region.category == "positive":
                crop = crop_positive(image, region)
                subdir = POSITIVE_SUBDIR
                subkind = ""
                conf = region.confidence if region.confidence is not None else 1.0
                fname = f"{image_stem}_l{region.id:02d}.png"
            else:  # negative
                crop = crop_negative(image, region)
                subdir = NEGATIVE_SUBDIR
                subkind = region.subkind or "background"
                conf = 1.0
                fname = f"{image_stem}_n{region.id:02d}.png"

            out_dir = output_dir / split / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / fname, format="PNG")

            records.append(CropRecord(
                split=split,
                source_image=image_stem,
                region_id=region.id,
                text=region.text,
                crop_relpath=f"{split}/{subdir}/{fname}",
                confidence=conf,
                category=region.category,
                subkind=subkind,
            ))
            if region.category == "positive":
                n_pos += 1
            else:
                n_neg += 1
        print(
            f"  - {image_stem} → {split}: {n_pos} pos / {n_neg} neg",
            file=sys.stderr,
        )

    _write_labels_tsv(output_dir / "labels.tsv", records)
    for split in split_map:
        split_records = [r for r in records if r.split == split]
        _write_labels_tsv(output_dir / split / "labels.tsv", split_records)

    counts: dict[str, int] = {}
    for split in split_map:
        srecs = [r for r in records if r.split == split]
        counts[split] = len(srecs)
        counts[f"{split}_pos"] = sum(1 for r in srecs if r.category == "positive")
        counts[f"{split}_neg"] = sum(1 for r in srecs if r.category == "negative")
    counts["_total"] = len(records)
    counts["_pos_total"] = sum(1 for r in records if r.category == "positive")
    counts["_neg_total"] = sum(1 for r in records if r.category == "negative")
    return counts


def _write_labels_tsv(path: Path, records: list[CropRecord]) -> None:
    """labels.tsv: filename text split source confidence category subkind の TSV。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(LABELS_TSV_HEADER)
        for r in records:
            w.writerow([
                r.crop_relpath, r.text, r.split, r.source_image,
                r.confidence, r.category, r.subkind,
            ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract recognition crops from annotations")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recognition"))
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="claude_verified=False の region も含める (デフォルトはスキップ)",
    )
    args = parser.parse_args(argv)

    if not args.samples_dir.is_dir():
        print(f"[extract_crops] samples not found: {args.samples_dir}", file=sys.stderr)
        return 1
    if not args.annotations_dir.is_dir():
        print(f"[extract_crops] annotations not found: {args.annotations_dir}", file=sys.stderr)
        return 1

    counts = extract_crops(
        args.samples_dir,
        args.annotations_dir,
        args.output_dir,
        require_verified=not args.include_unverified,
    )
    print(f"[extract_crops] done. counts: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
