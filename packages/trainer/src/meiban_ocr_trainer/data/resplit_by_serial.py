"""serial-disjoint な train/val/test 再分割(リーク排除)。

現状の split は画像/動画単位のため、同一シリアルが複数動画に跨って別 split に入り
test の 99.4% が train とシリアル重複(リーク)している。本ツールは統合 labels.tsv の
**split 列のみ**を「シリアル単位で完全分離」になるよう再計算し、新ファイルに書き出す。

- crop ファイルは移動しない(dataset.py は split 列でフィルタし root/filename で画像解決、
  両者は独立なので物理位置が train/real/ のままでも split=test として正しく読める)。
- 元 labels.tsv は不変。出力は labels_serial_split.tsv + 監査用 serial_split_map.yaml。
- positive はシリアル単位で disjoint 分割(層化=シリアル文字列でソートし比例貪欲割当)。
- negative(serial 無し)は source(image_stem)単位で比例割当(リークには無関係)。

使用例:
    python -m meiban_ocr_trainer.data.resplit_by_serial \
        --in-tsv data/recognition/labels.tsv \
        --out-tsv data/recognition/labels_serial_split.tsv \
        --map-out data/recognition/serial_split_map.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def _norm_serial(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _proportional_assign(keys: list[str], ratios: dict[str, float]) -> dict[str, str]:
    """keys を ratios に従って各 split に比例貪欲割当(決定的)。

    各ステップで「現在の充足率 (assigned/target) が最小の split」に割り当てることで、
    入力順(=層化のためソート済)に沿って各 split へ満遍なく散らす。
    """
    n = len(keys)
    target = {s: ratios[s] * n for s in SPLITS}
    assigned: dict[str, int] = {s: 0 for s in SPLITS}
    out: dict[str, str] = {}
    for k in keys:
        # 最も target に対して不足している split を選ぶ(tie は SPLITS 順で安定)
        best = min(SPLITS, key=lambda s: ((assigned[s] + 1) / target[s] if target[s] > 0 else float("inf"), SPLITS.index(s)))
        out[k] = best
        assigned[best] += 1
    return out


def resplit(in_tsv: Path, out_tsv: Path, map_out: Path, seed: int, ratios: dict[str, float]) -> dict:
    rows: list[dict] = []
    with in_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # --- positive: シリアル単位 ---
    serials_to_crops: dict[str, int] = {}
    for r in rows:
        if (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip():
            s = _norm_serial(r["text"])
            serials_to_crops[s] = serials_to_crops.get(s, 0) + 1
    # 層化: シリアル文字列でソート(隣接する類似シリアルが比例割当で3 splitに散る)
    serials_sorted = sorted(serials_to_crops.keys())
    serial_split = _proportional_assign(serials_sorted, ratios)

    # --- negative: source(image_stem)単位 ---
    neg_sources: set[str] = set()
    for r in rows:
        if not ((r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()):
            neg_sources.add(r.get("source") or "")
    neg_sorted = sorted(neg_sources)
    rng = random.Random(seed)
    rng.shuffle(neg_sorted)  # negative は層化不要、seed で決定的シャッフル
    source_split = _proportional_assign(neg_sorted, ratios)

    # --- split 列を書き換え ---
    for r in rows:
        is_pos = (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()
        if is_pos:
            r["split"] = serial_split[_norm_serial(r["text"])]
        else:
            r["split"] = source_split.get(r.get("source") or "", "train")

    # --- 自己検証: serial 集合の3 split 交差が空 ---
    by_split: dict[str, set] = {s: set() for s in SPLITS}
    for s, sp in serial_split.items():
        by_split[sp].add(s)
    assert not (by_split["train"] & by_split["val"]), "leak: train∩val"
    assert not (by_split["train"] & by_split["test"]), "leak: train∩test"
    assert not (by_split["val"] & by_split["test"]), "leak: val∩test"

    # --- 出力 ---
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    src_sha = hashlib.sha256(in_tsv.read_bytes()).hexdigest()[:16]
    map_data = {
        "seed": seed,
        "ratios": ratios,
        "source_tsv": str(in_tsv),
        "source_tsv_sha256_16": src_sha,
        "n_unique_serials": len(serials_sorted),
        "train": sorted(by_split["train"]),
        "val": sorted(by_split["val"]),
        "test": sorted(by_split["test"]),
    }
    map_out.parent.mkdir(parents=True, exist_ok=True)
    with map_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(map_data, f, allow_unicode=True, sort_keys=False)

    # --- 統計 ---
    stats: dict = {"by_split": {}}
    for sp in SPLITS:
        sp_rows = [r for r in rows if r["split"] == sp]
        pos = [r for r in sp_rows if (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()]
        neg = [r for r in sp_rows if r not in pos]
        stats["by_split"][sp] = {
            "serials": len(by_split[sp]),
            "crops": len(sp_rows),
            "positive": len(pos),
            "negative": len(neg),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="serial-disjoint な train/val/test 再分割")
    p.add_argument("--in-tsv", type=Path, default=Path("data/recognition/labels.tsv"))
    p.add_argument("--out-tsv", type=Path, default=Path("data/recognition/labels_serial_split.tsv"))
    p.add_argument("--map-out", type=Path, default=Path("data/recognition/serial_split_map.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=str, default="0.66,0.10,0.24",
                   help="train,val,test の serial 比率(カンマ区切り)")
    args = p.parse_args(argv)

    tr, va, te = (float(x) for x in args.ratios.split(","))
    ratios = {"train": tr, "val": va, "test": te}

    if not args.in_tsv.exists():
        print(f"ERROR: not found: {args.in_tsv}", file=sys.stderr)
        return 1

    stats = resplit(args.in_tsv, args.out_tsv, args.map_out, args.seed, ratios)
    print(f"[resplit] wrote {args.out_tsv}")
    print(f"[resplit] wrote {args.map_out}")
    for sp in SPLITS:
        s = stats["by_split"][sp]
        print(f"  {sp:5s}: serials={s['serials']:4d}  crops={s['crops']:6d}  "
              f"pos={s['positive']:6d}  neg={s['negative']:5d}")
    print("[resplit] serial 3-split 交差ゼロ: OK (assert 通過)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
