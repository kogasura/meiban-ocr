"""末尾 2nd-pass の効果測定: 右端 K セルを高解像で再読し pos10/11 を差し替える。

背景: 誤読の大半が pos10/11 に集中し、デコーダ変更 (fixed-head/attention) では
不変だった = 末尾の視覚特徴が 32×128 への縮小で潰れている疑い (encoder/入力側)。
full crop では末尾1文字 ≈ 10px 幅だが、右端4セルだけを 32×128 に再cropすれば
1文字 ≈ 32px と3倍の解像度になる。

手順 (1 crop あたり):
  1. full read (従来どおり 32×128)
  2. glyph_transplant の deskew + セル推定で 12 文字セルを得る
  3. 右端 K セル領域を再crop → 32×128 → 同じモデルで再読 (2nd pass)
  4. アンカー照合: 2nd pass の先頭 (K-2) 文字が full read の対応位置と一致した
     場合のみ、末尾2文字を 2nd pass の値で差し替え (誤適用ガード)
  5. EM / pos別誤りの before/after を比較

Usage:
    python -m meiban_ocr_trainer.tools.eval_tail_second_pass \\
        --model runs/cr_replaced/best.pt --root data/recognition_pad \\
        --labels data/recognition_pad/labels_serial_split.tsv --split test \\
        --tail-cells 4 --json runs/eval_tail2pass.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from meiban_ocr_trainer.data.glyph_transplant import (
    estimate_cell_bounds,
    estimate_skew_deg,
    find_serial_band,
    verify_mm_anchor,
    _ink_mask,
)
from meiban_ocr_trainer.tools.diagnose_pipeline import _decode_logits, normalize_crop
from meiban_ocr_trainer.tools.eval_recognition import load_predictor


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _char_timesteps(logits: np.ndarray, blank_idx: int = 36) -> list[int]:
    """CTC greedy path で emit された各文字の timestep を返す (長さ = 出力文字数)。"""
    idx = logits.argmax(axis=-1)
    out, prev = [], -1
    for t, c in enumerate(idx.tolist()):
        if c != blank_idx and c != prev:
            out.append(t)
        prev = c
    return out


def alignment_tail_crop(img: np.ndarray, logits: np.ndarray,
                        tail_cells: int) -> np.ndarray | None:
    """full read の CTC アライメントから右端 tail_cells 文字の領域を切る。

    deskew もセル推定も不要なので、古典推定が失敗する難 crop でも常に動く
    (cells モードは MM 検証で test の43%を弾き、修正対象の難 crop ごと捨てていた)。
    縦は全高 (傾き・隣接行込み — tail モデル側の訓練分布も同じ作り方にすること)。
    """
    ts = _char_timesteps(logits)
    if len(ts) != 12:
        return None
    T = logits.shape[0]
    w = img.shape[1]
    # 対象文字の手前の blank 区間中点あたりから右端まで
    t_start = ts[12 - tail_cells]
    t_prev = ts[12 - tail_cells - 1]
    x0 = int(round((t_start + t_prev) / 2 / T * w))
    if w - x0 < 8:
        return None
    return img[:, x0:]


def tail_crop(img: np.ndarray, tail_cells: int) -> np.ndarray | None:
    """deskew + セル推定で右端 tail_cells セルの帯領域を返す (水平化済み)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    deg = estimate_skew_deg(gray)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    desk = cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    gray_d = cv2.cvtColor(desk, cv2.COLOR_BGR2GRAY) if desk.ndim == 3 else desk
    band = find_serial_band(_ink_mask(gray_d))
    if band is None:
        return None
    bounds = estimate_cell_bounds(gray_d, band)
    if bounds is None:
        return None
    # セル推定ズレ = 「右端4セル」が実際は別の文字を含む = ラベル/置換ノイズ。
    # glyph_transplant と同じ MM アンカー (pos4/5='M') で検証し、不合格は捨てる。
    if not verify_mm_anchor(gray_d, band, bounds):
        return None
    y0, y1 = band
    pitch = (bounds[-1] - bounds[0]) / 12.0
    x0 = bounds[12 - tail_cells]
    # 右は少しだけ余白 (末尾文字の切れ防止)、上下も pitch の 1/4 だけ拡げる
    x1 = min(w, int(round(bounds[12] + pitch * 0.25)))
    pad_y = max(1, int(round(pitch * 0.25)))
    yy0 = max(0, y0 - pad_y)
    yy1 = min(h, y1 + 1 + pad_y)
    if x1 - x0 < 8 or yy1 - yy0 < 6:
        return None
    return desk[yy0:yy1, x0:x1]


def evaluate(model_path: Path, root: Path, labels: Path, split: str,
             tail_cells: int, limit: int,
             tail_model_path: Path | None = None,
             conf_thresholds: tuple[float, ...] = (0.0, 0.8, 0.9, 0.95, 0.98),
             loc_mode: str = "cells") -> dict:
    predict, mt, tok, resize_mode = load_predictor(model_path)
    # 末尾専用モデル (v10 は reject 訓練の副作用で部分シリアルを全列 blank と読むため、
    # 2nd pass には部分 crop で訓練した読み手が必要)
    if tail_model_path is not None:
        t_predict, t_mt, t_tok, t_resize = load_predictor(tail_model_path)
    else:
        t_predict, t_mt, t_tok, t_resize = predict, mt, tok, resize_mode

    rows = []
    with labels.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == split and (r.get("category") or "positive") == "positive" \
                    and (r.get("text") or "").strip():
                rows.append(r)
    if limit:
        rows = rows[:limit]

    anchor_len = tail_cells - 2
    samples: list[tuple[str, str, str | None, float]] = []  # (gt, full, tail, tail_conf)
    batch_imgs, metas = [], []

    def flush():
        if not batch_imgs:
            return
        full_arrs = [normalize_crop(im, resize_mode) for im, kind in batch_imgs if kind == "full"]
        full_logits = predict(full_arrs) if full_arrs else None
        full_dec = _decode_logits(mt, tok, full_logits) if full_arrs else []
        if loc_mode == "alignment":
            # full の CTC アライメントから tail crop を作る (この時点で初めて切れる)
            tail_arrs, owner = [], []
            for i, meta in enumerate(metas):
                tc = alignment_tail_crop(meta["img"], full_logits[i], tail_cells)
                if tc is not None:
                    tail_arrs.append(normalize_crop(tc, t_resize))
                    owner.append(i)
            tail_dec = _decode_logits(t_mt, t_tok, t_predict(tail_arrs)) if tail_arrs else []
            tails = {o: d for o, d in zip(owner, tail_dec)}
            for i, meta in enumerate(metas):
                full = _norm(full_dec[i][0])
                if i in tails:
                    samples.append((meta["gt"], full, _norm(tails[i][0]), float(tails[i][1])))
                else:
                    samples.append((meta["gt"], full, None, 0.0))
        else:
            tail_arrs = [normalize_crop(im, t_resize) for im, kind in batch_imgs if kind == "tail"]
            tail_dec = _decode_logits(t_mt, t_tok, t_predict(tail_arrs)) if tail_arrs else []
            fi = ti = 0
            for meta in metas:
                full = _norm(full_dec[fi][0]); fi += 1
                tail, tconf = None, 0.0
                if meta["has_tail"]:
                    tail = _norm(tail_dec[ti][0])
                    tconf = float(tail_dec[ti][1])
                    ti += 1
                samples.append((meta["gt"], full, tail, tconf))
        batch_imgs.clear()
        metas.clear()

    for r in rows:
        img = cv2.imread(str(root / r["filename"]))
        if img is None:
            continue
        gt = _norm(r["text"])
        if loc_mode == "alignment":
            batch_imgs.append((img, "full"))
            metas.append({"gt": gt, "img": img, "has_tail": False})
        else:
            tc = tail_crop(img, tail_cells)
            batch_imgs.append((img, "full"))
            if tc is not None:
                batch_imgs.append((tc, "tail"))
            metas.append({"gt": gt, "has_tail": tc is not None})
        if len(batch_imgs) >= 128:
            flush()
    flush()

    n = len(samples)
    em_base = sum(1 for gt, full, _, _ in samples if full == gt)
    pos_err_base: Counter = Counter()
    for gt, full, _, _ in samples:
        if len(full) == len(gt):
            for p, (a, b) in enumerate(zip(gt, full)):
                if a != b:
                    pos_err_base[p] += 1

    by_threshold = {}
    fix_examples, break_examples = [], []
    for th in conf_thresholds:
        em = 0
        pos_err: Counter = Counter()
        n_changed = n_fixed = n_broken = 0
        for gt, full, tail, tconf in samples:
            final = full
            if tail is not None and tconf >= th and len(full) == 12 \
                    and len(tail) == tail_cells \
                    and tail[:anchor_len] == full[12 - tail_cells:10]:
                cand = full[:10] + tail[anchor_len:]
                if cand != full:
                    n_changed += 1
                    if full != gt and cand == gt:
                        n_fixed += 1
                        if th == conf_thresholds[-1] and len(fix_examples) < 10:
                            fix_examples.append({"gt": gt, "full": full, "fixed": cand})
                    elif full == gt and cand != gt:
                        n_broken += 1
                        if th == conf_thresholds[-1] and len(break_examples) < 10:
                            break_examples.append({"gt": gt, "full": full, "broken": cand})
                final = cand
            em += final == gt
            if len(final) == len(gt):
                for p, (a, b) in enumerate(zip(gt, final)):
                    if a != b:
                        pos_err[p] += 1
        by_threshold[str(th)] = {
            "EM": round(em / n * 100, 2),
            "pos10": pos_err.get(10, 0), "pos11": pos_err.get(11, 0),
            "n_changed": n_changed, "n_fixed": n_fixed, "n_broken": n_broken,
        }

    n_geom_fail = sum(1 for _, _, t, _ in samples if t is None)
    return {
        "model": str(model_path),
        "tail_model": str(tail_model_path) if tail_model_path else None,
        "split": split, "tail_cells": tail_cells, "loc_mode": loc_mode,
        "n": n,
        "EM_base": round(em_base / n * 100, 2),
        "pos_errors_base": {str(k): v for k, v in sorted(pos_err_base.items())},
        "n_geom_fail": n_geom_fail,
        "by_threshold": by_threshold,
        "fix_examples": fix_examples,
        "break_examples": break_examples,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="末尾 2nd-pass の効果測定")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/recognition_pad"))
    p.add_argument("--labels", type=Path,
                   default=Path("data/recognition_pad/labels_serial_split.tsv"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--tail-cells", type=int, default=4)
    p.add_argument("--tail-model", type=Path, default=None,
                   help="末尾専用モデル (.pt)。未指定なら --model で末尾も読む")
    p.add_argument("--loc-mode", choices=["cells", "alignment"], default="cells",
                   help="末尾の位置特定: cells=deskew+セル推定 / alignment=CTCアライメント")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    res = evaluate(args.model, args.root, args.labels, args.split,
                   args.tail_cells, args.limit, args.tail_model,
                   loc_mode=args.loc_mode)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("fix_examples", "break_examples")},
                     ensure_ascii=False, indent=2))
    print("fixed例:", json.dumps(res["fix_examples"][:5], ensure_ascii=False))
    print("broken例:", json.dumps(res["break_examples"][:5], ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
