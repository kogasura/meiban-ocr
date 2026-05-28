"""合成 negative 生成 (Ericsson pattern 外のランダム英数字 + 純背景)。

2 種類を生成:
- **other_text**: Ericsson regex に一致しないランダム英数字を背景に描画。
  「文字はあるが対象コードではない」ケースをモデルに学ばせる。
- **background**: 背景 crop を 32×128 にリサイズして保存 (文字無し)。
  「テクスチャだけある領域」のサンプル。

両方とも labels.tsv では text="", category=negative。subkind は other_text / background。

設計判断:
- ランダム英数字の長さは 5〜12 字、Ericsson に類似 (大文字+数字、`E`/`M` を含むことあり)
  だが strict regex `/^E[39]\\d{2}MM\\d{6}$/` には**絶対に一致しない**よう構造を変える:
  - prefix が "E3" or "E9" でないか、または "MM" が異なる位置にあるか
  - 簡単のため、最初の 2 文字を E[39] でなくすか、MM を含まないかでフィルタする
- ratio パラメータで other_text:background の比率制御。デフォルト 70:30
  (other_text の方が hard negative としての価値が高い)。

Usage:
    python -m meiban_ocr_trainer.data.synth_negative \\
        --backgrounds-dir data/backgrounds \\
        --output-dir data/recognition/train/synth_neg \\
        --labels-tsv data/recognition/labels.tsv \\
        --count 1000 --other-text-ratio 0.7
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import string
import sys
from pathlib import Path

import cv2
import numpy as np

from meiban_ocr_trainer.data.synth_positive import (
    TARGET_H,
    TARGET_W,
    _prep_background,
    render_synth_positive,
)
from meiban_ocr_trainer.data.text_replace import DEFAULT_FONTS

ALPHANUM = string.digits + string.ascii_uppercase
ERICSSON_STRICT = re.compile(r"^E[39]\d{2}MM\d{6}$")


def generate_non_ericsson_text(rng: random.Random) -> str:
    """Ericsson strict regex に**絶対一致しない**ランダム英数字を返す。

    戦略:
    - 長さ 5〜15 字 (Ericsson は 12 字なので短い/長いも混ぜる)
    - 各文字は ALPHANUM から uniform sample
    - 生成後に regex 一致を rejection sample (確率的にほぼ毎回 1 回で通る)
    """
    for _ in range(20):  # safety limit
        length = rng.randint(5, 15)
        text = "".join(rng.choices(ALPHANUM, k=length))
        if not ERICSSON_STRICT.match(text):
            return text
    # 極めて稀: 安全側に倒してハイフン入りを返す (Ericsson regex は - を含まない)
    return "X-" + "".join(rng.choices(ALPHANUM, k=10))


def render_background_only(
    bg: np.ndarray, rng: random.Random,
) -> np.ndarray:
    """背景 crop を 32×128 にリサイズして返す (テキスト描画なし)。"""
    canvas = _prep_background(bg, TARGET_W, TARGET_H, rng)
    # 軽い blur + noise だけ (positive 側と統計を揃える)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
    noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_synth_negatives(
    backgrounds_dir: Path,
    output_dir: Path,
    labels_tsv: Path,
    count: int,
    other_text_ratio: float = 0.7,
    font_paths: list[str] | None = None,
    seed: int = 4242,
) -> dict[str, int]:
    """`count` 件の synth negative を生成。

    Returns:
        {"other_text": N1, "background": N2, "total": N1+N2}
    """
    font_paths = [p for p in (font_paths or DEFAULT_FONTS) if Path(p).exists()]
    if not font_paths:
        raise RuntimeError("No usable fonts found.")
    bg_paths = sorted(backgrounds_dir.glob("*.png"))
    if not bg_paths:
        raise RuntimeError(f"No backgrounds found in {backgrounds_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    np.random.seed(seed)

    n_other_text = int(count * other_text_ratio)
    n_background = count - n_other_text

    new_records: list[tuple[str, str]] = []  # (relpath, subkind)

    # other_text: ランダム英数字を描画
    for i in range(n_other_text):
        bg = cv2.imread(str(rng.choice(bg_paths)), cv2.IMREAD_COLOR)
        if bg is None:
            continue
        text = generate_non_ericsson_text(rng)
        img = render_synth_positive(bg, text, font_paths, rng)
        fname = f"synth_n_other_{i:06d}.png"
        cv2.imwrite(str(output_dir / fname), img)
        new_records.append((f"train/synth_neg/{fname}", "other_text"))

    # background: 純背景リサイズ
    for i in range(n_background):
        bg = cv2.imread(str(rng.choice(bg_paths)), cv2.IMREAD_COLOR)
        if bg is None:
            continue
        img = render_background_only(bg, rng)
        fname = f"synth_n_bg_{i:06d}.png"
        cv2.imwrite(str(output_dir / fname), img)
        new_records.append((f"train/synth_neg/{fname}", "background"))

    # labels.tsv に append
    with labels_tsv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for relpath, subkind in new_records:
            w.writerow([relpath, "", "train", "synth_neg", 1.0, "negative", subkind])

    counts = {
        "other_text": sum(1 for _, s in new_records if s == "other_text"),
        "background": sum(1 for _, s in new_records if s == "background"),
        "total": len(new_records),
    }
    print(
        f"[synth_negative] wrote {counts['total']} images "
        f"(other_text: {counts['other_text']}, background: {counts['background']})",
        file=sys.stderr,
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic negative generator")
    parser.add_argument("--backgrounds-dir", type=Path, default=Path("data/backgrounds"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/recognition/train/synth_neg"),
    )
    parser.add_argument(
        "--labels-tsv", type=Path, default=Path("data/recognition/labels.tsv"),
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--other-text-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args(argv)

    if not args.backgrounds_dir.is_dir():
        print(f"[synth_negative] backgrounds not found: {args.backgrounds_dir}", file=sys.stderr)
        return 1
    if not args.labels_tsv.exists():
        print(f"[synth_negative] labels.tsv not found: {args.labels_tsv}", file=sys.stderr)
        return 1

    counts = generate_synth_negatives(
        args.backgrounds_dir, args.output_dir, args.labels_tsv,
        count=args.count, other_text_ratio=args.other_text_ratio,
        seed=args.seed,
    )
    print(f"[synth_negative] done: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
