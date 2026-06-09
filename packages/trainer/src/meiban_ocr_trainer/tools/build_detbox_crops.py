"""det-box 由来の認識訓練 crop を生成する(訓練/推論の crop 分布ミスマッチ対策)。

E2E 実測で「det-box 上の認識EM 34%(人手 text_bbox crop では 90%)」と判明。原因は訓練 crop が
人手 text_bbox(+4px tight) なのに推論は paddle det box(unclip1.6, しばしば多行混入)を食う
分布ミスマッチ。本スクリプトは **paddle det を回して得た box そのもの**を crop にし、
GT シリアルと対応付けて認識器を「推論で来る crop」で訓練できるデータを作る。

マッチング:
- det box が覆う GT serial(text_bbox) を containment=交差/GT面積 で判定。
- box 内に「mostly inside(>=thresh)」な GT serial が **ちょうど1つ** → positive(そのシリアルで label)。
- 0 個 → negative(非シリアル box, text="")。1画像あたり --neg-per-image で上限。
- 2 個以上(多行混入) → 曖昧なので skip。
split は serial_split_map.yaml の serial→split を継承(serial-disjoint 維持)。negative は train。

Usage:
    python -m meiban_ocr_trainer.tools.build_detbox_crops \\
        --det-model dist-uranus2/model/paddle/ppocrv4_det.onnx --det-long-side 960 \\
        --samples-dir samples --annotations-dir annotations \\
        --split-map data/recognition_pad/serial_split_map.yaml \\
        --output-dir data/recognition_detcrop --neg-per-image 4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from meiban_ocr_trainer.tools.mine_nonserial_negatives import db_postprocess, preprocess_for_det


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _containment(box, gt) -> float:
    ix1, iy1 = max(box[0], gt[0]), max(box[1], gt[1])
    ix2, iy2 = min(box[2], gt[2]), min(box[3], gt[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(1, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    return inter / gt_area


def _load_split_map(path: Path) -> dict[str, str]:
    d = yaml.safe_load(path.read_text())
    out = {}
    for split in ("train", "val", "test"):
        for s in (d.get(split) or []):
            out[_norm(s)] = split
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="det-box 由来の認識訓練 crop 生成")
    p.add_argument("--det-model", type=Path, default=Path("dist-uranus2/model/paddle/ppocrv4_det.onnx"))
    p.add_argument("--det-long-side", type=int, default=960)
    p.add_argument("--samples-dir", type=Path, default=Path("samples"))
    p.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    p.add_argument("--split-map", type=Path, default=Path("data/recognition_pad/serial_split_map.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("data/recognition_detcrop"))
    p.add_argument("--containment", type=float, default=0.5)
    p.add_argument("--max-height-ratio", type=float, default=1.6,
                   help="box高 <= この倍率 × serial text高 のみ positive(多行label-block除外)")
    p.add_argument("--neg-per-image", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    split_map = _load_split_map(args.split_map)
    sess = ort.InferenceSession(str(args.det_model), providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    out_dir = args.output_dir
    for sp in ("train", "val", "test"):
        (out_dir / sp).mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {"pos": {"train": 0, "val": 0, "test": 0}, "neg": 0,
              "skip_multi": 0, "skip_multiline_block": 0}
    n_images = 0

    for af in sorted(args.annotations_dir.glob("img_*.json")):
        d = json.loads(af.read_text())
        gt = [
            (r.get("text_bbox") or r.get("bbox"), _norm(r.get("text") or ""))
            for r in d.get("regions", [])
            if (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()
            and (r.get("text_bbox") or r.get("bbox"))
        ]
        if not gt:
            continue
        img_path = args.samples_dir / (d.get("image") or (af.stem + ".jpg"))
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        n_images += 1
        if args.limit and n_images > args.limit:
            n_images -= 1
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        seg = sess.run([out_name], {in_name: preprocess_for_det(rgb, args.det_long_side)})[0]
        boxes = db_postprocess(np.asarray(seg)[0, 0].astype(np.float32), oh, ow)

        neg_budget = args.neg_per_image
        for bi, box in enumerate(boxes):
            # box 内に mostly inside な GT serial とその text高
            inside = [(serial, _containment(box, gbox), gbox[3] - gbox[1])
                      for gbox, serial in gt]
            inside = [(s, c, gh) for s, c, gh in inside if c >= args.containment]
            x1, y1, x2, y2 = box
            box_h = y2 - y1
            if x2 - x1 < 4 or box_h < 4:
                continue
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if len(inside) == 1:
                serial, _c, gt_h = inside[0]
                # 多行 label-block(box が serial 行より十分高い)は除外
                if box_h > args.max_height_ratio * max(1, gt_h):
                    counts["skip_multiline_block"] += 1
                    continue
                split = split_map.get(serial)
                if split is None:
                    continue  # serial が split-map に無い(=対象外)
                fn = f"{split}/{af.stem}_b{bi}.png"
                cv2.imwrite(str(out_dir / fn), crop)
                rows.append([fn, serial, split, af.stem, "1.0", "positive", "detbox"])
                counts["pos"][split] += 1
            elif len(inside) == 0:
                if neg_budget <= 0:
                    continue
                neg_budget -= 1
                fn = f"train/{af.stem}_b{bi}_neg.png"
                cv2.imwrite(str(out_dir / fn), crop)
                rows.append([fn, "", "train", af.stem, "1.0", "negative", "det_nonserial"])
                counts["neg"] += 1
            else:
                counts["skip_multi"] += 1

    labels_path = out_dir / "labels.tsv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "text", "split", "source", "confidence", "category", "subkind"])
        w.writerows(rows)

    print(json.dumps({
        "images": n_images, "rows": len(rows),
        "pos": counts["pos"], "neg": counts["neg"],
        "skip_multi_serial": counts["skip_multi"],
        "skip_multiline_block": counts["skip_multiline_block"],
        "labels": str(labels_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
