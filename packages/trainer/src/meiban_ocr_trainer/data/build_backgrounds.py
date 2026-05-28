"""背景ライブラリ構築。`data/backgrounds/` に subkind=background の crop を集める。

TRDG ベースの合成 positive (B2) と合成 negative (B3) で、自然な背景の上にテキストを
レンダリングするための素材。labels.tsv (v2) を参照し、`subkind=background` の負例 crop
だけを抽出して `data/backgrounds/` にコピーする。

設計判断:
- **既存の negative crop を再利用**: わざわざ samples から再 crop するのではなく、
  既に `subkind=background` として人手検証された crop をそのまま使う。重複作業を避ける。
- **コピー (symlink ではない)**: TRDG/Albumentations が画像を破壊的に開かないとはいえ、
  別工程で resize/save される場合に元 crop を壊さないようコピー。

Usage:
    python -m meiban_ocr_trainer.data.build_backgrounds \\
        [--labels-tsv data/recognition/labels.tsv] \\
        [--output-dir data/backgrounds]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def collect_backgrounds(
    labels_tsv: Path,
    crops_root: Path,
    output_dir: Path,
) -> int:
    """labels.tsv から subkind=background 行を抽出し、crop を output_dir へコピー。

    Args:
        labels_tsv: data/recognition/labels.tsv
        crops_root: crop 画像のルート (labels.tsv の filename 列はこのディレクトリからの相対パス)
        output_dir: 出力先 (作られていなければ作成)

    Returns:
        コピーされた枚数。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with labels_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("category") != "negative":
                continue
            if row.get("subkind") != "background":
                continue
            src = crops_root / row["filename"]
            if not src.exists():
                print(f"  ! missing: {src}", file=sys.stderr)
                continue
            # ファイル名衝突を避けるため source 画像 stem を prefix に
            dst_name = f"{row['source']}_{Path(row['filename']).name}"
            dst = output_dir / dst_name
            shutil.copy2(src, dst)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build background library from negative crops")
    parser.add_argument(
        "--labels-tsv", type=Path, default=Path("data/recognition/labels.tsv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/backgrounds"),
    )
    args = parser.parse_args(argv)

    if not args.labels_tsv.exists():
        print(f"[build_backgrounds] labels.tsv not found: {args.labels_tsv}", file=sys.stderr)
        return 1

    crops_root = args.labels_tsv.parent  # labels.tsv と同じディレクトリが crop ルート
    n = collect_backgrounds(args.labels_tsv, crops_root, args.output_dir)
    print(f"[build_backgrounds] copied {n} background crops to {args.output_dir}",
          file=sys.stderr)
    if n == 0:
        print(
            "[build_backgrounds] WARNING: no backgrounds found. "
            "Run extract_crops.py first to produce negative crops with subkind=background.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
