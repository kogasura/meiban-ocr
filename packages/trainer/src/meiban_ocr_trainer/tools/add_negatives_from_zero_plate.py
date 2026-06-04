"""0-plate frame に negative region を追加する (KGI reject 訓練データ生成)。

RapidOCR で 0 plate が出た frame は、 (a) モーションブラー、 (b) 反射、 (c) 撮影中の
画面遷移、 (d) RapidOCR 自体の検出漏れ、 のいずれか。 (a)(b)(c) は実運用で
「読めない → 検出無し」を学習させたい negative。 (d) は誤って negative 化されるが、
全体の少数派 (目視で 10-20% 程度) なので統計的に許容する方針。

各 zero-plate frame に対し、 一定数のランダム negative bbox を追加する:
  - フレームサイズ 1920x1080 想定
  - bbox サイズ ≒ 128x32 の plate スケール ± バリエーション
  - frame の上下左右にランダム配置 (中心は plate 多い領域なので軽く避ける、 ただし可)
  - 1 frame あたり default 5 個

Usage:
    python -m meiban_ocr_trainer.tools.add_negatives_from_zero_plate
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def add_negatives_to_annotation(
    ann_path: Path,
    n_per_frame: int = 5,
    bbox_w_range: tuple[int, int] = (100, 200),
    bbox_h_range: tuple[int, int] = (28, 60),
    seed: int | None = None,
) -> int:
    """0-plate annotation に n_per_frame 個の negative bbox を追加。

    Returns: 追加した negative region 数。
    """
    rng = random.Random(seed if seed is not None else hash(ann_path.stem))
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    regions = data.get("regions", [])

    # 既に positive がある or negative が既に追加されているなら skip
    if any(r.get("category", "positive") == "positive" for r in regions):
        return 0
    if any(r.get("category") == "negative" for r in regions):
        return 0

    img_w, img_h = data.get("image_size", [1920, 1080])

    next_id = max((r.get("id", -1) for r in regions), default=-1) + 1
    added = 0
    for _ in range(n_per_frame):
        bw = rng.randint(*bbox_w_range)
        bh = rng.randint(*bbox_h_range)
        if bw >= img_w or bh >= img_h:
            continue
        x1 = rng.randint(0, img_w - bw)
        y1 = rng.randint(0, img_h - bh)
        regions.append({
            "id": next_id,
            "category": "negative",
            "bbox": [x1, y1, x1 + bw, y1 + bh],
            "subkind": "background",
            "claude_verified": True,  # auto-generated 但し reject 訓練として用いるので verified 相当
        })
        next_id += 1
        added += 1

    data["regions"] = regions
    ann_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add negative regions to zero-plate annotations.",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("annotations"),
    )
    parser.add_argument(
        "--n-per-frame", type=int, default=5,
        help="1 frame あたりの negative bbox 数 (default 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.annotations_dir.is_dir():
        print(f"annotations-dir not found: {args.annotations_dir}")
        return 1

    total_added = 0
    n_files = 0
    for p in sorted(args.annotations_dir.glob("img_*.json")):
        n = add_negatives_to_annotation(p, args.n_per_frame, seed=args.seed)
        if n > 0:
            n_files += 1
            total_added += n
    print(f"added {total_added} negatives across {n_files} zero-plate frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
