"""末尾 2nd-pass 用データセット生成: 右端 K セル crop + 末尾 K 文字ラベル。

v10 (full-serial CRNN) は reject 訓練の副作用で部分シリアルを全列 blank と読むため、
末尾再読には専用の読み手が必要。本ツールは recognition_pad の positive crop から
deskew + セル推定 (glyph_transplant と同一・検証済み) で右端 K セルを切り出し、
`data/recognition_tail/{split}/` + labels.tsv を生成する。

- split は元 tsv の split をそのまま継承 (serial-disjoint 性を保存)
- train は replaced (グリフ移植) 行も含む = 末尾の見た目多様性をそのまま利用
- セル推定に失敗した crop はスキップ (品質 > 量)

Usage:
    python -m meiban_ocr_trainer.tools.build_tail_crops \\
        --root data/recognition_pad --labels labels_serial_split_replaced.tsv \\
        --out-root data/recognition_tail --tail-cells 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from meiban_ocr_trainer.tools.eval_tail_second_pass import (
    alignment_tail_crop,
    tail_crop,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="末尾 crop データセット生成")
    p.add_argument("--root", type=Path, default=Path("data/recognition_pad"))
    p.add_argument("--labels", type=str, default="labels_serial_split_replaced.tsv")
    p.add_argument("--out-root", type=Path, default=Path("data/recognition_tail"))
    p.add_argument("--tail-cells", type=int, default=4)
    p.add_argument("--loc-mode", choices=["cells", "alignment"], default="cells",
                   help="cells=deskew+セル推定 (MM検証で難cropを棄却) / "
                        "alignment=full モデルの CTC アライメント (全crop で動く。"
                        "評価/本番と同じ分布で訓練するならこちら)")
    p.add_argument("--align-model", type=Path, default=None,
                   help="alignment モードで使う full モデル (.pt)")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    predict = resize_mode = None
    if args.loc_mode == "alignment":
        from meiban_ocr_trainer.tools.diagnose_pipeline import normalize_crop
        from meiban_ocr_trainer.tools.eval_recognition import load_predictor
        if args.align_model is None:
            raise SystemExit("--loc-mode alignment requires --align-model")
        predict, _mt, _tok, resize_mode = load_predictor(args.align_model)

    rows = list(csv.DictReader((args.root / args.labels).open(encoding="utf-8"),
                               delimiter="\t"))
    out_rows = []
    n_in = n_ok = n_skip = 0
    for r in rows:
        if (r.get("category") or "positive") != "positive":
            continue
        text = (r.get("text") or "").strip()
        if len(text) != 12:
            continue
        n_in += 1
        if args.limit and n_in > args.limit:
            break
        img = cv2.imread(str(args.root / r["filename"]))
        if img is None:
            n_skip += 1
            continue
        if args.loc_mode == "alignment":
            from meiban_ocr_trainer.tools.diagnose_pipeline import normalize_crop
            logits = predict([normalize_crop(img, resize_mode)])[0]
            tc = alignment_tail_crop(img, logits, args.tail_cells)
        else:
            tc = tail_crop(img, args.tail_cells)
        if tc is None:
            n_skip += 1
            continue
        split = r["split"]
        rel = Path(split) / (Path(r["filename"]).stem + "_tail.png")
        out_path = args.out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), tc)
        out_rows.append({
            "filename": str(rel),
            "text": text[-args.tail_cells:],
            "split": split,
            "source": r.get("source", ""),
            "confidence": r.get("confidence", ""),
            "category": "positive",
            "subkind": r.get("subkind", ""),
        })
        n_ok += 1
        if n_ok % 5000 == 0:
            print(f"[build_tail_crops] {n_ok} done...")

    fieldnames = ["filename", "text", "split", "source", "confidence", "category", "subkind"]
    with (args.out_root / "labels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    from collections import Counter
    print(f"[build_tail_crops] in={n_in} ok={n_ok} skip={n_skip}")
    print("[build_tail_crops] split:", dict(Counter(r['split'] for r in out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
