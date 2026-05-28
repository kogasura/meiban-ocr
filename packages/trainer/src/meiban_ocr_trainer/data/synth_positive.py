"""合成 positive 生成 (Ericsson serial × 背景ライブラリ)。

`data/backgrounds/` の背景 crop をキャンバスにして、ランダム生成した Ericsson serial
を PIL でレンダリングし、`data/recognition/train/synth_pos/` に出力する。

text_replace.py が「実画像のテキストを書き換える」のに対し、本スクリプトは「実画像の
背景に新たに serial を載せる」 → 大量生成に向く (背景多様性 × serial 全空間)。

設計判断:
- **TRDG を使わない**: 依存を増やさず text_replace.py と同じ PIL + フォント資産を再利用。
  perspective/rotation は Albumentations (既存依存) で訓練時に追加する想定。
- **背景 resize**: 元 background crop は様々なサイズ。アスペクト比保持で短辺
  ~40px にリサイズしてから 128x32 にランダム crop/pad してから描画。
- **augmentation はここでは入れない**: 訓練 transform 側 (augment.py) に任せる。
  ここは "クリーンな合成画像" を出力する責務。

Usage:
    python -m meiban_ocr_trainer.data.synth_positive \\
        --backgrounds-dir data/backgrounds \\
        --output-dir data/recognition/train/synth_pos \\
        --labels-tsv data/recognition/labels.tsv \\
        --count 2000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from meiban_ocr_trainer.data.text_replace import (
    DEFAULT_FONTS,
    generate_random_ericsson_serial,
)

# 出力画像サイズ (CRNN 入力 32×128 に合わせる)
TARGET_W = 128
TARGET_H = 32


def _prep_background(
    bg: np.ndarray, target_w: int, target_h: int, rng: random.Random,
) -> np.ndarray:
    """背景 crop をアスペクト保持で短辺 target_h に揃え、横方向にランダム crop/pad。

    Why: 背景は様々な解像度・アスペクト比なので、まず縦をそろえる → 横は
    target_w 以上なら crop、未満なら mirror padding。serial の描画前にこの正規化を行う。
    """
    h, w = bg.shape[:2]
    # scale: 短辺を target_h にする (アスペクト保持)
    scale = target_h / h
    new_w = max(target_w, int(w * scale))
    bg2 = cv2.resize(bg, (new_w, target_h), interpolation=cv2.INTER_AREA)
    if bg2.shape[1] > target_w:
        x0 = rng.randint(0, bg2.shape[1] - target_w)
        bg2 = bg2[:, x0:x0 + target_w]
    elif bg2.shape[1] < target_w:
        pad = target_w - bg2.shape[1]
        bg2 = cv2.copyMakeBorder(
            bg2, 0, 0, pad // 2, pad - pad // 2,
            cv2.BORDER_REFLECT_101,
        )
    return bg2


def _sample_text_color(bg: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    """背景の平均輝度の補色寄り (明暗反転) を text 色として返す。BGR。

    Why: 背景が暗ければ明るい文字、明るければ暗い文字を選ぶことで可読性を確保。
    text_replace.py と同様の発想だが、こちらは「中央付近の最暗値を採取」できないので
    平均輝度から逆方向にジッタする。
    """
    mean = bg.reshape(-1, 3).mean(axis=0)  # BGR
    gray = float(0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2])
    # 背景が暗いなら 200〜240 / 明るいなら 20〜60
    if gray < 128:
        v = rng.randint(200, 245)
        return (v, v, v)
    v = rng.randint(10, 60)
    return (v, v, v)


def _fit_font(font_path: str, text: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    """text が (max_w, max_h) に収まる最大フォントサイズ。"""
    lo, hi = 6, max(10, max_h * 2)
    best = ImageFont.truetype(font_path, lo)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = f.getbbox(text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= max_w and h <= max_h:
            best = f
            lo = mid
        else:
            hi = mid - 1
    return best


def render_synth_positive(
    bg: np.ndarray,
    text: str,
    font_paths: list[str],
    rng: random.Random,
) -> np.ndarray:
    """背景に Ericsson serial を描画した 32×128 BGR 画像を返す。"""
    canvas = _prep_background(bg, TARGET_W, TARGET_H, rng)
    text_color = _sample_text_color(canvas, rng)
    font_path = rng.choice(font_paths)
    # 描画領域に 10% マージン
    inner_w = int(TARGET_W * 0.90)
    inner_h = int(TARGET_H * 0.85)
    font = _fit_font(font_path, text, inner_w, inner_h)
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    cx = (TARGET_W - text_w) // 2
    cy = (TARGET_H - text_h) // 2
    draw.text(
        (cx - bbox[0], cy - bbox[1]),
        text,
        font=font,
        fill=(text_color[2], text_color[1], text_color[0]),  # BGR→RGB
    )
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    # 軽い blur + noise で合成感を抑える (augmentation は訓練側に任せるので最小限のみ)
    out = cv2.GaussianBlur(out, (3, 3), 0.5)
    noise = np.random.normal(0, 3, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def generate_synth_positives(
    backgrounds_dir: Path,
    output_dir: Path,
    labels_tsv: Path,
    count: int,
    font_paths: list[str] | None = None,
    seed: int = 1337,
) -> int:
    """`count` 件の synth positive を生成し、labels.tsv に append。

    Returns:
        実際に生成した件数。
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

    new_records: list[tuple[str, str]] = []
    for i in range(count):
        bg_path = rng.choice(bg_paths)
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg is None:
            print(f"  ! skip unreadable bg: {bg_path}", file=sys.stderr)
            continue
        text = generate_random_ericsson_serial(rng)
        img = render_synth_positive(bg, text, font_paths, rng)
        fname = f"synth_p_{i:06d}.png"
        cv2.imwrite(str(output_dir / fname), img)
        relpath = f"train/synth_pos/{fname}"
        new_records.append((relpath, text))

    # labels.tsv に append (v2 schema: 7 列)
    with labels_tsv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for relpath, text in new_records:
            w.writerow([relpath, text, "train", "synth_pos", 1.0, "positive", ""])

    print(
        f"[synth_positive] wrote {len(new_records)} images using {len(bg_paths)} backgrounds × {len(font_paths)} fonts",
        file=sys.stderr,
    )
    return len(new_records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic positive generator (Ericsson)")
    parser.add_argument("--backgrounds-dir", type=Path, default=Path("data/backgrounds"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/recognition/train/synth_pos"),
    )
    parser.add_argument(
        "--labels-tsv", type=Path, default=Path("data/recognition/labels.tsv"),
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    if not args.backgrounds_dir.is_dir():
        print(f"[synth_positive] backgrounds not found: {args.backgrounds_dir}", file=sys.stderr)
        return 1
    if not args.labels_tsv.exists():
        print(f"[synth_positive] labels.tsv not found: {args.labels_tsv}", file=sys.stderr)
        return 1

    n = generate_synth_positives(
        args.backgrounds_dir, args.output_dir, args.labels_tsv,
        count=args.count, seed=args.seed,
    )
    print(f"[synth_positive] done: {n} images", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
