"""テキスト書き換え水増し。HANDOFF.md §4 Step 3 を実装。

実画像クロップ (data/recognition/train/real/*.png) のテキスト部分を
ランダム生成した別の Ericsson serial に置き換え、data/recognition/train/replaced/ に出力。

各 real 画像から N variants 生成 → 文字バランスを統計的に均一化して訓練多様性を確保。

Why: 実画像で集まる serial は連番なので文字バランスに偏りが出る
(E303MM5xxxxx ばかり)。書き換えで全数字 0-9 + A-Z (Ericsson は限定字種だが) を
均等出現させる。

実装:
1. crop を読む。背景色をクロップ4隅からサンプリング。テキスト色 (≈黒) を中央付近からサンプリング。
2. テキスト領域全体を cv2.inpaint で背景化。
3. 新 serial を PIL でレンダリング、テキスト境界に fit するフォントサイズを自動決定。
4. 軽い Gaussian blur + ノイズで馴染ませる。
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

# Linux標準のmonospaceフォント + Windows由来の Consolas
DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
    "/mnt/c/Windows/Fonts/consola.ttf",
    "/mnt/c/Windows/Fonts/CascadiaMono.ttf",
]


def generate_random_ericsson_serial(rng: random.Random | None = None) -> str:
    """Dummy 範囲 `^E300MM\\d{6}$` のみを生成 (Iter5 v5 #8 fix)。

    Why: 旧版は `E[39]\\d{2}MM\\d{6}` 全空間 (2×10⁸) から一様抽出していたが、
    本番 Ericsson serial も同じ空間に存在する (300K件、衝突確率 0.15%) ため、
    1000枚の synth 生成で確率的に ~1-2 件の実シリアル衝突が起きうる。
    `vendors.ts` で「実 serial = E[39]\\d{2}MM\\d{6}」が定義されており、
    dummy 専用に `E300MM\\d{6}` を予約する SECURITY.md 方針と整合させる。

    Trade-off: prefix 多様性 (2 × 10² = 200 種) を捨てるが、suffix の 10⁶ 種で
    文字バランス学習には十分。実際の本番でモデルが見るのは E[39]\\d{2}MM\\d{6}
    全空間だが、prefix 4 字 "E300" は固定でもデコーダの CTC 学習を破壊しない
    (suffix 部分のみ可変なので char-level の多様性は維持される)。
    """
    rng = rng or random.Random()
    suffix = "".join(rng.choices("0123456789", k=6))
    return f"E300MM{suffix}"


def _sample_background_color(img: np.ndarray) -> tuple[int, int, int]:
    """画像の4隅 (各 4x4 ピクセル) の平均を背景色として採用。"""
    h, w = img.shape[:2]
    s = 4
    corners = [
        img[:s, :s], img[:s, w - s:], img[h - s:, :s], img[h - s:, w - s:],
    ]
    mean = np.mean([c.reshape(-1, 3).mean(axis=0) for c in corners], axis=0)
    return tuple(int(x) for x in mean)  # BGR


def _sample_text_color(img: np.ndarray) -> tuple[int, int, int]:
    """画像中央部の最低輝度近くのピクセル平均をテキスト色として採用。"""
    h, w = img.shape[:2]
    center = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    # 暗い側 20%ile 程度のピクセルを使う
    th = np.percentile(gray, 20)
    mask = gray < th
    if mask.sum() < 10:
        return (20, 20, 20)
    dark_pixels = center[mask]
    mean = dark_pixels.mean(axis=0)
    return tuple(int(x) for x in mean)


def _fit_font_size(font_path: str, text: str, target_w: int, target_h: int) -> tuple[ImageFont.FreeTypeFont, tuple[int, int]]:
    """target box にちょうど収まるフォントサイズを二分探索で決める。"""
    lo, hi = 6, max(10, target_h * 2)
    best = ImageFont.truetype(font_path, lo)
    best_size = best.getbbox(text)[2:4]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = f.getbbox(text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target_w and h <= target_h:
            best = f
            best_size = (w, h)
            lo = mid
        else:
            hi = mid - 1
    return best, best_size


def text_replace(
    crop: np.ndarray,
    new_text: str,
    font_paths: list[str],
    rng: random.Random | None = None,
) -> np.ndarray:
    """クロップ画像のテキストを new_text に置き換えた画像を返す。"""
    rng = rng or random.Random()
    h, w = crop.shape[:2]
    img = crop.copy()

    # Why: text 領域は概ねクロップ全体。上下に少しだけ縦マージンを取り背景をサンプル可能に。
    text_y1 = max(0, int(h * 0.10))
    text_y2 = min(h, int(h * 0.90))
    text_x1 = max(0, int(w * 0.02))
    text_x2 = min(w, int(w * 0.98))

    bg_color = _sample_background_color(img)
    text_color = _sample_text_color(img)

    # Inpaint テキスト領域
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[text_y1:text_y2, text_x1:text_x2] = 255
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # フォントを1つ選んで、target box にちょうど fit させる
    font_path = rng.choice(font_paths)
    target_w = text_x2 - text_x1
    target_h = text_y2 - text_y1
    font, (rw, rh) = _fit_font_size(font_path, new_text, int(target_w * 0.95), int(target_h * 0.95))

    # PIL で描画
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    cx = (w - rw) // 2
    cy = (h - rh) // 2
    # PIL の座標は bbox の左上を起点に描かない (フォント内部の baseline オフセットあり) のでgetbbox補正
    bbox = font.getbbox(new_text)
    draw.text((cx - bbox[0], cy - bbox[1]), new_text, font=font,
              fill=(text_color[2], text_color[1], text_color[0]))  # BGR→RGB
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 軽い blur + ノイズで馴染ませる
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    labels_tsv: Path,
    variants_per_crop: int = 50,
    font_paths: list[str] | None = None,
    seed: int = 42,
) -> int:
    """train/real/ の全 PNG に対し variants_per_crop 個の書き換え版を train/replaced/ に出力。"""
    font_paths = [p for p in (font_paths or DEFAULT_FONTS) if Path(p).exists()]
    if not font_paths:
        raise RuntimeError("No usable fonts found. Place TTF in fonts/ or install monospace fonts.")
    print(f"[text_replace] using {len(font_paths)} fonts", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    pngs = sorted(input_dir.glob("*.png"))
    new_records: list[tuple[str, str]] = []  # (relpath, text)

    for src in pngs:
        crop = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if crop is None:
            print(f"  - {src.name}: failed to read, skip", file=sys.stderr)
            continue
        for i in range(variants_per_crop):
            new_serial = generate_random_ericsson_serial(rng)
            out_img = text_replace(crop, new_serial, font_paths, rng)
            out_name = f"{src.stem}_v{i:03d}.png"
            cv2.imwrite(str(output_dir / out_name), out_img)
            relpath = f"train/replaced/{out_name}"
            new_records.append((relpath, new_serial))

    # labels.tsv に追記 (v2 schema: filename text split source confidence category subkind)
    # 既存ファイルが v1 (5列) ならヘッダ末尾は category/subkind 列が無いが、DictReader
    # が許容するため append のみで両立できる。
    with labels_tsv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for relpath, text in new_records:
            w.writerow([relpath, text, "train", "replaced", 1.0, "positive", ""])

    print(
        f"[text_replace] wrote {len(new_records)} variants from {len(pngs)} real crops",
        file=sys.stderr,
    )
    return len(new_records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text-replace augmentation")
    parser.add_argument("--input-dir", type=Path, default=Path("data/recognition/train/real"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recognition/train/replaced"))
    parser.add_argument("--labels-tsv", type=Path, default=Path("data/recognition/train/labels.tsv"))
    parser.add_argument("--all-labels-tsv", type=Path, default=Path("data/recognition/labels.tsv"))
    parser.add_argument("--variants", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"[text_replace] input not found: {args.input_dir}", file=sys.stderr)
        return 1

    n = augment_dataset(
        args.input_dir, args.output_dir, args.labels_tsv,
        variants_per_crop=args.variants, seed=args.seed,
    )
    # ルートの labels.tsv にも反映
    if args.all_labels_tsv.exists() and n > 0:
        import shutil
        # train/labels.tsv をルートに再構成する代わりに、append のみ
        with args.labels_tsv.open("r", encoding="utf-8") as f:
            train_rows = list(csv.reader(f, delimiter="\t"))
        # 追加行 (header除く新規分) を all_labels_tsv に append
        new_rows = [r for r in train_rows[1:] if "/replaced/" in r[0]]
        with args.all_labels_tsv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            for row in new_rows:
                w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
