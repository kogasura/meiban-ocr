"""Stage 1 (samples/ + annotations/*.json) → Stage 2 (data/recognition/{train,val,test}) 変換。

各 label の `bbox` (パディング済みのラベル領域) を切り出し、PNG + labels.tsv を生成。
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

config.yaml で split を上書きすることも可 (将来の動画フレーム対応用)。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# v0 デフォルト分割 (画像単位)
DEFAULT_SPLIT: dict[str, set[str]] = {
    "train": {"img_001", "img_003"},
    "val": {"img_002"},
    "test": {"img_004"},
}


@dataclass
class CropRecord:
    split: str
    source_image: str
    label_id: int
    text: str
    crop_relpath: str  # labels.tsv に書く相対パス (例: train/real/img_001_l03.png)
    confidence: float


def _resolve_split(image_stem: str, split_map: dict[str, set[str]]) -> str | None:
    for split, stems in split_map.items():
        if image_stem in stems:
            return split
    return None


def crop_label(
    image: Image.Image,
    text_bbox: list[int],
    margin_px: int = 4,
) -> Image.Image:
    """text_bbox を少しだけ padding してクロップ (CRNN 1行テキスト認識用)。

    Why text_bbox instead of bbox: CRNN+CTC は単一行テキスト認識器であり、複数行が写った
    クロップを学習させると CTC の整合がとれない (ground truth は serial 12文字なのに画像
    には RRU 22F3 / シリアル / 日付 / 会社名 の4行が見える)。
    text_bbox は serial だけを囲んでおり、CRNN のトレーニング・推論の両方に整合する
    (runtime 検出器も text-line を返す設計)。
    """
    x1, y1, x2, y2 = text_bbox
    w, h = image.size
    return image.crop((
        max(0, x1 - margin_px),
        max(0, y1 - margin_px),
        min(w, x2 + margin_px),
        min(h, y2 + margin_px),
    ))


def extract_crops(
    samples_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    split_map: dict[str, set[str]] = None,
    require_verified: bool = True,
) -> dict[str, int]:
    """Stage 1 → Stage 2 変換のメインルーチン。

    Args:
        samples_dir: 元画像ディレクトリ
        annotations_dir: annotations/*.json
        output_dir: data/recognition (この下に train/val/test を切る)
        split_map: image_stem → split の対応。未指定なら DEFAULT_SPLIT
        require_verified: True の場合、claude_verified=False の label をスキップ

    Returns:
        各 split のサンプル数辞書
    """
    split_map = split_map or DEFAULT_SPLIT
    for split in split_map:
        (output_dir / split / "real").mkdir(parents=True, exist_ok=True)

    records: list[CropRecord] = []

    for ann_path in sorted(annotations_dir.glob("img_*.json")):
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)
        image_stem = Path(ann["image"]).stem
        split = _resolve_split(image_stem, split_map)
        if split is None:
            print(f"  - {image_stem}: no split assigned, skip", file=sys.stderr)
            continue

        image_path = samples_dir / ann["image"]
        if not image_path.exists():
            print(f"  - {image_stem}: missing image {image_path}", file=sys.stderr)
            continue

        image = Image.open(image_path).convert("RGB")

        n_kept = 0
        for label in ann["labels"]:
            if require_verified and not label.get("claude_verified", False):
                continue
            text_bbox = label["text_bbox"]
            crop = crop_label(image, text_bbox)
            label_id = label["id"]
            fname = f"{image_stem}_l{label_id:02d}.png"
            out_path = output_dir / split / "real" / fname
            crop.save(out_path, format="PNG")
            records.append(
                CropRecord(
                    split=split,
                    source_image=image_stem,
                    label_id=label_id,
                    text=label["text"],
                    crop_relpath=f"{split}/real/{fname}",
                    confidence=label["confidence"],
                )
            )
            n_kept += 1
        print(f"  - {image_stem} → {split}: {n_kept} crops", file=sys.stderr)

    # labels.tsv をまとめて書き出し (全 split 統合 + 各 split 個別)
    _write_labels_tsv(output_dir / "labels.tsv", records)
    for split in split_map:
        split_records = [r for r in records if r.split == split]
        _write_labels_tsv(output_dir / split / "labels.tsv", split_records)

    counts = {split: sum(1 for r in records if r.split == split) for split in split_map}
    counts["_total"] = len(records)
    return counts


def _write_labels_tsv(path: Path, records: list[CropRecord]) -> None:
    """labels.tsv: filename\\ttext\\tsplit\\tsource\\tconfidence の TSV。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["filename", "text", "split", "source", "confidence"])
        for r in records:
            w.writerow([r.crop_relpath, r.text, r.split, r.source_image, r.confidence])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract recognition crops from annotations")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recognition"))
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="claude_verified=False のlabelも含める (デフォルトはスキップ)",
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
