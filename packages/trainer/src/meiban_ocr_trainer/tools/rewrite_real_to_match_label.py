"""real positive crop の画像内テキストをラベル GT に合わせて書き換える (one-shot)。

問題: security sanitize で annotations のテキストフィールドだけ ダミー化 (e.g.,
`E305MM503786` → `E300MM000013`) したが、**画像内のピクセルは実シリアルのまま**。
結果、ラベルと画像が不整合になり、訓練時にモデルが学習困難:
  - 画像: `E301MM004014` (実)
  - GT:   `E300MM000013` (ダミー)
  - 損失: 「E301... を見たら E300... を出せ」と教えている → モデル混乱

本スクリプトは `text_replace` を使って **画像内のテキスト領域を inpaint + 再描画**
し、画像 = GT になるよう揃える。train/val/test の real positive 全件に適用。

副作用:
- 物理的真値性 (本物の銘板の質感) は失われる (inpaint で背景再生成)
- ただし `train/replaced/` の 1900 件は既に同じ手法で生成されているため、訓練分布
  との整合性は向上する
- 配布物への実シリアル漏洩リスクが消える (画像内に実シリアル無し)

Usage:
    python -m meiban_ocr_trainer.tools.rewrite_real_to_match_label
    # 全 splits の real positive crop を in-place で書き換え
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

from meiban_ocr_trainer.data.text_replace import DEFAULT_FONTS, text_replace


def rewrite_all(
    labels_tsv: Path,
    data_root: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, int]:
    """labels.tsv を読み、real positive crop を全件書き換え。

    Returns:
        {"rewritten": N, "skipped": M, "missing": K}
    """
    rng = random.Random(seed)
    fonts = [p for p in DEFAULT_FONTS if Path(p).exists()]
    if not fonts:
        raise RuntimeError("No usable fonts found.")

    with labels_tsv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    n_rewritten = 0
    n_skipped = 0
    n_missing = 0
    n_total_candidates = sum(
        1 for r in rows
        if r["category"] == "positive"
        and "real/" in r["filename"]
        and r["source"].startswith("img_")
    )
    print(f"[rewrite] target: {n_total_candidates} real positive crops", file=sys.stderr)

    for i, row in enumerate(rows):
        # 対象: positive かつ filename に "real/" を含み (synth_pos/replaced 除外)、
        # source が img_XXX (実画像由来) のもの
        if row["category"] != "positive":
            n_skipped += 1
            continue
        if "real/" not in row["filename"]:
            n_skipped += 1
            continue
        if not row["source"].startswith("img_"):
            n_skipped += 1
            continue

        crop_path = data_root / row["filename"]
        if not crop_path.exists():
            print(f"  ! missing: {crop_path}", file=sys.stderr)
            n_missing += 1
            continue

        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            print(f"  ! unreadable: {crop_path}", file=sys.stderr)
            n_missing += 1
            continue

        gt_text = row["text"]
        if not gt_text:
            n_skipped += 1
            continue

        if dry_run:
            print(
                f"  [DRY] would rewrite {crop_path.name} ({crop.shape[1]}×{crop.shape[0]}) "
                f"→ '{gt_text}'",
                file=sys.stderr,
            )
            n_rewritten += 1
            continue

        # text_replace で画像内テキスト領域を gt_text に書き換え
        new_img = text_replace(crop, gt_text, fonts, rng)
        cv2.imwrite(str(crop_path), new_img)
        n_rewritten += 1
        if n_rewritten % 10 == 0:
            print(f"  ... {n_rewritten}/{n_total_candidates}", file=sys.stderr)

    return {"rewritten": n_rewritten, "skipped": n_skipped, "missing": n_missing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite real positive crops to match dummy GT labels.",
    )
    parser.add_argument(
        "--labels-tsv", type=Path, default=Path("data/recognition/labels.tsv"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/recognition"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="actual rewriting なし、何件対象かだけ出す",
    )
    args = parser.parse_args(argv)

    if not args.labels_tsv.exists():
        print(f"[rewrite] labels.tsv not found: {args.labels_tsv}", file=sys.stderr)
        return 1

    counts = rewrite_all(args.labels_tsv, args.data_root, args.seed, args.dry_run)
    print(f"\n[rewrite] done: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
