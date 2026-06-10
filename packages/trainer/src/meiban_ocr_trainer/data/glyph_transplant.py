"""グリフ移植による書き換え水増し (text-replacement augmentation v2)。

同一 crop 内の数字セル (文字1個ぶんのストリップ) を別の数字セルのコピーで
置き換え、ラベルを更新した replaced crop を生成する。

Why glyph transplant (旧 text_replace の inpaint+フォント描画は使わない):
- 置換元も置換先も同じ crop 内なので、照明・ボケ・金属質感・遠近が保存される。
  inpaint やフォント描画の artifact による訓練分布の歪みが原理的に出ない。
- 訓練 151 シリアルは pos0-9 がほぼ定数で pos10/11 のみ多様 (本番誤読の95%が
  pos10/11 に集中 = 末尾桁のデータ枯渇)。移植は「この撮影条件でこの数字が末尾に
  来た見た目」を実画素で作るため、末尾桁の見た目多様性に直撃する。

幾何 (重要): crop 内のシリアル行はしばしば斜めに走る (回転撮影) ため、垂直
ストリップの単純コピーでは文字が縦にズレて崩壊する。手順:
  1. projection-profile 法で deskew 角を推定 (±14° 走査、行プロファイルの
     尖り最大の角)。
  2. 水平化空間でシリアル帯 (行プロファイルの最密帯) と文字セル境界を推定。
  3. 帯の行範囲だけでセルを移植 (隣接行は触らない)。
  4. 逆回転で元の傾き空間へ書き戻し、変更セル近傍のみ alpha 合成
     (二重リサンプリングを変更箇所に限定)。

置換可能位置: Ericsson strict `E[39]\\d{2}MM\\d{6}` の数字位置のうち
pos1 を除く {2, 3, 6, 7, 8, 9, 10, 11} (pos1 は 3|9 制約があるため触らない)。

Usage:
    python -m meiban_ocr_trainer.data.glyph_transplant \\
        --root data/recognition_pad --labels labels_serial_split.tsv \\
        --variants 2 --seed 42 --out-labels labels_serial_split_replaced.tsv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

SERIAL_LEN = 12
# E[39]\d{2}MM\d{6} の数字位置 (pos1 は 3|9 制約のため除外)
DIGIT_POSITIONS = (2, 3, 6, 7, 8, 9, 10, 11)
# 置換確率: 末尾4桁 (本番誤読の95%) を重点的に
REPLACE_PROB = {2: 0.3, 3: 0.3, 6: 0.3, 7: 0.3, 8: 0.8, 9: 0.8, 10: 0.8, 11: 0.8}
FEATHER_PX = 2
MIN_CROP_W = 48
MAX_SKEW_DEG = 14


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """テキスト画素 (暗) の 0/1 マスク。Otsu 二値化。"""
    _, th = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def estimate_skew_deg(gray: np.ndarray) -> float:
    """projection-profile 法: 行プロファイルの尖り (分散) が最大になる回転角。"""
    ink = _ink_mask(gray).astype(np.float32)
    h, w = ink.shape
    best_deg, best_score = 0.0, -1.0
    for deg in range(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        rot = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        prof = rot.sum(axis=1)
        score = float((prof ** 2).sum())
        if score > best_score:
            best_score, best_deg = score, float(deg)
    return best_deg


def find_serial_band(ink: np.ndarray) -> tuple[int, int] | None:
    """水平化済み ink の行プロファイルから最密のテキスト帯 (y0, y1) を返す。"""
    prof = ink.sum(axis=1).astype(np.float32)
    if prof.max() <= 0:
        return None
    peak = int(np.argmax(prof))
    th = prof.max() * 0.35
    y0 = peak
    while y0 > 0 and prof[y0 - 1] >= th:
        y0 -= 1
    y1 = peak
    while y1 < len(prof) - 1 and prof[y1 + 1] >= th:
        y1 += 1
    if y1 - y0 < 5:
        return None
    pad = max(1, (y1 - y0) // 6)
    return max(0, y0 - pad), min(len(prof) - 1, y1 + pad)


def estimate_cell_bounds(
    gray_deskewed: np.ndarray, band: tuple[int, int]
) -> list[int] | None:
    """シリアル帯内の列プロファイルから 12 文字セル境界 (13 個) を推定。

    1. ink 列プロファイルをギャップで glyph run に分割。ちょうど 12 run なら
       ギャップ中点を境界に採用 (最も正確)。
    2. それ以外 (ブラーで結合 / 断片化 / 隣接行の写り込みで余分な ink) は
       (x0, pitch) のグリッド探索で「セル中央に ink・境界に非 ink」を最大化。
    最後に MM アンカー検証 (verify_mm_anchor) を呼び出し側で行うこと。
    """
    y0, y1 = band
    ink = _ink_mask(gray_deskewed[y0 : y1 + 1])
    col = ink.sum(axis=0).astype(np.float32)
    w = len(col)
    nz = np.flatnonzero(col > 0)
    if len(nz) == 0 or int(nz[-1]) - int(nz[0]) < MIN_CROP_W:
        return None

    # --- 1. ギャップ分割 ---
    th = max(1.0, col.max() * 0.08)
    on = col > th
    runs: list[tuple[int, int]] = []
    i = 0
    while i < w:
        if on[i]:
            j = i
            while j + 1 < w and on[j + 1]:
                j += 1
            runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    # 幅が中央値の 1/4 未満の run はノイズとして除去
    if runs:
        med_w = float(np.median([b - a for a, b in runs]))
        runs = [(a, b) for a, b in runs if (b - a) >= med_w * 0.25]
    if len(runs) == SERIAL_LEN:
        bounds = [runs[0][0]]
        for k in range(1, SERIAL_LEN):
            bounds.append((runs[k - 1][1] + runs[k][0]) // 2)
        bounds.append(runs[-1][1])
        return bounds

    # --- 2. グリッド探索 ---
    x_lo, x_hi = int(nz[0]), int(nz[-1]) + 1
    extent = x_hi - x_lo
    best, best_score = None, -1e9
    csum = np.concatenate([[0.0], np.cumsum(col)])

    def seg_sum(a: float, b: float) -> float:
        ai = min(max(int(round(a)), 0), w)
        bi = min(max(int(round(b)), 0), w)
        if bi <= ai:
            return 0.0
        return float(csum[bi] - csum[ai])

    for pitch_scale in np.linspace(0.85, 1.08, 12):
        pitch = extent / SERIAL_LEN * pitch_scale
        if pitch < 3:
            continue
        for x0f in np.linspace(x_lo - pitch * 0.5, x_lo + pitch * 0.7, 13):
            score = 0.0
            for k in range(SERIAL_LEN):
                c0 = x0f + k * pitch
                # セル中央 60% に ink があるほど高く、境界 ±12% に ink があるほど低い
                score += seg_sum(c0 + pitch * 0.2, c0 + pitch * 0.8)
                score -= 2.0 * seg_sum(c0 - pitch * 0.12, c0 + pitch * 0.12)
            score -= 2.0 * seg_sum(x0f + SERIAL_LEN * pitch - pitch * 0.12,
                                   x0f + SERIAL_LEN * pitch + pitch * 0.12)
            # グリッド外に ink が残るのはペナルティ (12 文字で全 ink を覆うべき)
            score -= 1.5 * (seg_sum(0, x0f) + seg_sum(x0f + SERIAL_LEN * pitch, w))
            if score > best_score:
                best_score, best = score, (x0f, pitch)
    if best is None:
        return None
    x0f, pitch = best
    bounds = [int(round(x0f + k * pitch)) for k in range(SERIAL_LEN + 1)]
    bounds = [min(max(b, 0), w) for b in bounds]
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            return None
    return bounds


def verify_mm_anchor(
    gray_deskewed: np.ndarray, band: tuple[int, int], bounds: list[int],
    min_ncc: float = 0.55,
) -> bool:
    """セル整合の検証: pos4/pos5 は常に 'M' なので両セルの NCC が高いはず。

    セルが 1 文字ぶんズレていると pos4='M' vs pos5='5' 等になり相関が落ちる。
    ズレたままの移植はラベルノイズ (画像とラベルの不一致) になるため必ず弾く。
    """
    y0, y1 = band
    a = gray_deskewed[y0 : y1 + 1, bounds[4] : bounds[5]].astype(np.float32)
    b = gray_deskewed[y0 : y1 + 1, bounds[5] : bounds[6]].astype(np.float32)
    if a.size == 0 or b.size == 0:
        return False
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return False
    return float((a * b).sum() / denom) >= min_ncc


def _transplant_strip(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    band: tuple[int, int],
    bounds: list[int],
    dst_pos: int,
    src_pos: int,
    touched: np.ndarray,
) -> None:
    """帯の行範囲だけ src セルを dst セルへコピー (dst_img を in-place 変更、横 feather)。

    src_img は **無変更の元画像** を渡すこと。変更中の画像をドナーにすると、
    先に置換済みのセルから新グリフをコピーしてしまいラベル不一致になる。
    """
    y0, y1 = band
    dx0, dx1 = bounds[dst_pos], bounds[dst_pos + 1]
    sx0, sx1 = bounds[src_pos], bounds[src_pos + 1]
    dw = dx1 - dx0
    src = src_img[y0 : y1 + 1, sx0:sx1].copy()
    if src.shape[1] != dw:
        src = cv2.resize(src, (dw, src.shape[0]), interpolation=cv2.INTER_LINEAR)
    f = min(FEATHER_PX, dw // 3)
    dst_region = dst_img[y0 : y1 + 1, dx0:dx1]
    if f > 0:
        alpha = np.ones(dw, dtype=np.float32)
        ramp = (np.arange(1, f + 1) / (f + 1)).astype(np.float32)
        alpha[:f] = ramp
        alpha[-f:] = ramp[::-1]
        a = alpha[None, :, None] if dst_img.ndim == 3 else alpha[None, :]
        blended = src.astype(np.float32) * a + dst_region.astype(np.float32) * (1 - a)
        dst_img[y0 : y1 + 1, dx0:dx1] = blended.astype(dst_img.dtype)
    else:
        dst_img[y0 : y1 + 1, dx0:dx1] = src
    touched[y0 : y1 + 1, dx0:dx1] = 1.0


def make_variant(
    img: np.ndarray,
    text: str,
    rng: random.Random,
    forbidden: set[str],
    max_tries: int = 8,
) -> tuple[np.ndarray, str] | None:
    """1 variant 生成。幾何推定に失敗 / 置換不成立 / 禁止シリアル衝突なら None。"""
    if len(text) != SERIAL_LEN:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    if w < MIN_CROP_W:
        return None

    deg = estimate_skew_deg(gray)
    center = (w / 2, h / 2)
    m_fwd = cv2.getRotationMatrix2D(center, deg, 1.0)
    m_inv = cv2.getRotationMatrix2D(center, -deg, 1.0)
    desk = cv2.warpAffine(
        img, m_fwd, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    gray_d = cv2.cvtColor(desk, cv2.COLOR_BGR2GRAY) if desk.ndim == 3 else desk
    band = find_serial_band(_ink_mask(gray_d))
    if band is None:
        return None
    bounds = estimate_cell_bounds(gray_d, band)
    if bounds is None:
        return None
    # セルズレ = ラベルノイズの源。MM アンカーで検証できない crop は捨てる。
    if not verify_mm_anchor(gray_d, band, bounds):
        return None

    for _ in range(max_tries):
        plan: list[tuple[int, int]] = []
        chars = list(text)
        for dst in DIGIT_POSITIONS:
            if rng.random() >= REPLACE_PROB[dst]:
                continue
            donors = [s for s in DIGIT_POSITIONS if s != dst and text[s] != text[dst]]
            if not donors:
                continue
            src = rng.choice(donors)
            plan.append((dst, src))
            chars[dst] = text[src]
        new_text = "".join(chars)
        if not plan or new_text == text or new_text in forbidden:
            continue

        desk2 = desk.copy()
        touched = np.zeros((h, w), dtype=np.float32)
        for dst, src in plan:
            _transplant_strip(desk, desk2, band, bounds, dst, src, touched)

        # 逆回転で元の傾き空間へ。変更セル近傍のみ alpha 合成して
        # 二重リサンプリングの影響を変更箇所に限定する。
        back = cv2.warpAffine(
            desk2, m_inv, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        alpha = cv2.warpAffine(touched, m_inv, (w, h), flags=cv2.INTER_LINEAR)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        a3 = alpha[..., None] if img.ndim == 3 else alpha
        out = (
            back.astype(np.float32) * a3 + img.astype(np.float32) * (1 - a3)
        ).astype(img.dtype)
        return out, new_text
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="グリフ移植による書き換え水増し")
    p.add_argument("--root", type=Path, default=Path("data/recognition_pad"))
    p.add_argument("--labels", type=str, default="labels_serial_split.tsv")
    p.add_argument("--out-labels", type=str, default="labels_serial_split_replaced.tsv")
    p.add_argument("--out-subdir", type=str, default="train/replaced")
    p.add_argument("--variants", type=int, default=2, help="real positive 1枚あたりの生成数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="入力 crop 数上限 (0=全件、デバッグ用)")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    labels_path = args.root / args.labels
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8"), delimiter="\t"))

    # 生成ラベルが val/test シリアルへ衝突するとリーク (train で見た文字列が test に
    # 存在する) になるため禁止集合にする。train シリアルとの衝突は実在の組合せなので許容。
    forbidden = {
        r["text"].strip()
        for r in rows
        if r["split"] in ("val", "test") and (r.get("category") or "positive") == "positive"
    }

    out_dir = args.root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    n_in = n_gen = n_skip = 0
    for r in rows:
        if r["split"] != "train" or (r.get("category") or "positive") != "positive":
            continue
        text = (r["text"] or "").strip()
        if len(text) != SERIAL_LEN:
            continue
        n_in += 1
        if args.limit and n_in > args.limit:
            break
        img = cv2.imread(str(args.root / r["filename"]))
        if img is None:
            n_skip += 1
            continue
        stem = Path(r["filename"]).stem
        for k in range(args.variants):
            v = make_variant(img, text, rng, forbidden)
            if v is None:
                n_skip += 1
                continue
            out_img, new_text = v
            fname = f"{args.out_subdir}/{stem}_r{k}.png"
            cv2.imwrite(str(args.root / fname), out_img)
            new_rows.append({
                "filename": fname,
                "text": new_text,
                "split": "train",
                "source": r.get("source", ""),
                "confidence": r.get("confidence", ""),
                "category": "positive",
                "subkind": "replaced",
            })
            n_gen += 1

    out_path = args.root / args.out_labels
    fieldnames = ["filename", "text", "split", "source", "confidence", "category", "subkind"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        for r in new_rows:
            w.writerow(r)

    print(f"[glyph_transplant] in={n_in} generated={n_gen} skipped={n_skip}")
    print(f"[glyph_transplant] wrote {out_path} ({len(rows) + len(new_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
