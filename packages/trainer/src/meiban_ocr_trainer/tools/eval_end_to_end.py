"""End-to-end 実測: paddle det 検出 → custom CRNN 認識 を held-out フル画像で評価。

これまでの eval_recognition は「人手 text_bbox crop を所与にした認識器単体」を測っていた。
本ハーネスは本番(URANUS2 ハイブリッド)と同じく **paddle det が出す box を認識器に渡す**
end-to-end を測り、本番性能 = (検出カバー率) × (det-box 上の認識EM) を分解して出す。

- 検出: mine_nonserial_negatives の deployment 一致 det (detLongSide=736, unclip=1.6 等)。
- 認識: CRNN .pt (torch)。onnxruntime 1.26 は CRNN onnx で segfault するため .pt を使う
  (重みは onnx と同一)。crop_and_normalize / decode は eval_recognition と共有。

held-out: labels_serial_split.tsv の test serial(serial-disjoint)に属す positive region のみ採点。

Usage:
    python -m meiban_ocr_trainer.tools.eval_end_to_end \\
        --crnn runs/cr_pad_stretch/best.pt \\
        --det dist-uranus2/model/paddle/ppocrv4_det.onnx \\
        --samples-dir samples --annotations-dir annotations \\
        --labels data/recognition_pad/labels_serial_split.tsv --split test \\
        --limit 0 --json runs/eval_e2e.json
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

from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits,
    crop_and_normalize,
    normalize_crop,
)
from meiban_ocr_trainer.tools.eval_recognition import load_predictor
from meiban_ocr_trainer.tools.mine_nonserial_negatives import (
    db_postprocess,
    db_postprocess_quad,
    get_rotate_crop,
    preprocess_for_det,
)
from meiban_ocr_trainer.vendors import ERICSSON

STRICT = ERICSSON.strict_regex


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _containment(box, gt) -> float:
    """intersection / gt_area: box が GT serial 領域をどれだけ覆うか (0-1)。"""
    ix1, iy1 = max(box[0], gt[0]), max(box[1], gt[1])
    ix2, iy2 = min(box[2], gt[2]), min(box[3], gt[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(1, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    return inter / gt_area


def _test_serials(labels_path: Path, split: str) -> set[str]:
    out = set()
    with labels_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == split and (r.get("category") or "positive") == "positive":
                t = (r.get("text") or "").strip()
                if t:
                    out.add(_norm(t))
    return out


def evaluate(crnn_pt: Path, det_model: Path, samples_dir: Path, annotations_dir: Path,
             labels_path: Path, split: str, cover_thresh: float, limit: int,
             det_long_side: int = 736, box_mode: str = "rect",
             unclip_ratio: float = 1.6,
             tail_model: Path | None = None, tail_conf: float = 0.95,
             sim_app_preprocess: str = "none") -> dict:
    test_serials = _test_serials(labels_path, split)
    predict, mt, tok, resize_mode = load_predictor(crnn_pt)
    # 末尾 2nd-pass (quad モード専用): 12文字 read の box に対し、CTC アライメントで
    # 末尾4文字領域を再 crop → 専用モデルで再読 → アンカー一致 & conf>=tail_conf で
    # 末尾2文字を差し替える。clean test 実測 +1.1〜1.3pt (eval_tail_second_pass)。
    t_predict = t_tok = t_mt = None
    t_resize = resize_mode
    if tail_model is not None:
        t_predict, t_mt, t_tok, t_resize = load_predictor(tail_model)
    sess = ort.InferenceSession(str(det_model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    ann_files = sorted(annotations_dir.glob("img_*.json"))

    # 集計
    n_gt = 0            # held-out test-serial 領域の総数
    n_covered = 0       # det box が覆った数 (検出カバー)
    n_e2e_correct = 0   # 覆い かつ rec exact
    n_rec_correct_given_covered = 0
    n_det_boxes = 0
    n_box_serial_fire = 0   # regex 適合を吐いた box 数 (発火)
    n_box_false_fire = 0    # regex 適合だが画像内のどの GT serial とも不一致 (捏造)
    cover_box_ar = []   # 覆った det box のアスペクト比
    n_images = 0
    miss_examples = []  # 検出ミス例
    rec_fail_examples = []  # 覆ったが誤読の例
    # GT シリアルの原画素高 (text_bbox 高) 別の検出カバー/認識 集計
    from collections import defaultdict
    h_bins = [(0, 20), (20, 35), (35, 55), (55, 1_000_000)]

    def _hbin(h):
        for lo, hi in h_bins:
            if lo <= h < hi:
                return f"{lo}-{hi if hi < 1_000_000 else 'inf'}px"
        return "?"

    by_h = defaultdict(lambda: {"n": 0, "cov": 0, "e2e": 0})

    for af in ann_files:
        d = json.loads(af.read_text())
        regions = d.get("regions", [])
        gt_regions = [
            (r.get("text_bbox") or r.get("bbox"), _norm(r.get("text") or ""))
            for r in regions
            if (r.get("category") or "positive") == "positive"
            and _norm(r.get("text") or "") in test_serials
            and (r.get("text_bbox") or r.get("bbox"))
        ]
        if not gt_regions:
            continue
        img_name = d.get("image") or (af.stem + ".jpg")
        img_path = samples_dir / img_name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        n_images += 1
        if limit and n_images > limit:
            n_images -= 1
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sim_scale = 1
        if sim_app_preprocess != "none":
            # URANUS2 preprocessForOcr の再現: Rec.601 グレースケール + 2倍アップサンプル
            # (+ binarize 時は Otsu 二値化)。実機の入力分布での E2E を測るため。
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # cv2 は Rec.601 係数
            if sim_app_preprocess == "binarize2x":
                _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if (g == 0).mean() > 0.5:
                    g = 255 - g
            g = cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2),
                           interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
            sim_scale = 2  # det box は 2x 座標になるので GT も 2x で照合する
        oh, ow = rgb.shape[:2]
        seg = sess.run([out_name], {in_name: preprocess_for_det(rgb, det_long_side)})[0]
        seg_map = np.asarray(seg)[0, 0].astype(np.float32)
        if box_mode == "quad":
            quads = db_postprocess_quad(seg_map, oh, ow, unclip_ratio=unclip_ratio)
            # 被覆判定は GT bbox と同じ axis-aligned 外接矩形で行う
            boxes = [
                [int(q[:, 0].min()), int(q[:, 1].min()),
                 int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))]
                for q in quads
            ]
        else:
            quads = None
            boxes = db_postprocess(seg_map, oh, ow)
        n_det_boxes += len(boxes)

        # 全 GT serial (test+train) を発火判定の正解集合に
        all_gt_serials = {
            _norm(r.get("text") or "")
            for r in regions
            if (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()
        }

        # 全 det box を rec (発火率 + 各 GT に対する被覆/認識判定に使う)
        if boxes:
            if box_mode == "quad":
                from meiban_ocr_trainer.constants import INPUT_HEIGHT, INPUT_WIDTH
                empty = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
                crops = [get_rotate_crop(rgb, q) for q in quads]
                arrs = [normalize_crop(c, resize_mode) if c is not None else empty
                        for c in crops]
            else:
                arrs = [crop_and_normalize(rgb, b, resize_mode) for b in boxes]
            logits = predict(arrs)
            decoded = _decode_logits(mt, tok, logits)  # [(text,conf)]
            box_texts = [_norm(t) for t, _ in decoded]
            if t_predict is not None and box_mode == "quad":
                from meiban_ocr_trainer.tools.eval_tail_second_pass import (
                    alignment_tail_crop,
                )
                t_arrs, owners = [], []
                for bi, txt in enumerate(box_texts):
                    if len(txt) != 12 or crops[bi] is None:
                        continue
                    tc = alignment_tail_crop(crops[bi], logits[bi], 4)
                    if tc is not None:
                        t_arrs.append(normalize_crop(tc, t_resize))
                        owners.append(bi)
                if t_arrs:
                    t_dec = _decode_logits(t_mt, t_tok, t_predict(t_arrs))
                    for o, (tt, tc_conf) in zip(owners, t_dec):
                        tt = _norm(tt)
                        full = box_texts[o]
                        if (len(tt) == 4 and float(tc_conf) >= tail_conf
                                and tt[:2] == full[8:10]):
                            box_texts[o] = full[:10] + tt[2:]
        else:
            box_texts = []

        for bt in box_texts:
            if STRICT.match(bt):
                n_box_serial_fire += 1
                if bt not in all_gt_serials:
                    n_box_false_fire += 1

        for gt_box, gt_text in gt_regions:
            n_gt += 1
            gt_h = gt_box[3] - gt_box[1]
            if sim_scale != 1:
                gt_box = [c * sim_scale for c in gt_box]
            hb = _hbin(gt_h)
            by_h[hb]["n"] += 1
            # 本番に忠実: det は全 box を rec して serial 集合を出力する。
            # GT を覆う(containment>=thresh) box が1つでもあれば covered、
            # そのうち**どれか1つでも** rec==GT なら recognized(本番が正しく吐ける)。
            overlapping = [(i, _containment(b, gt_box)) for i, b in enumerate(boxes)]
            overlapping = [(i, c) for i, c in overlapping if c >= cover_thresh]
            if overlapping:
                n_covered += 1
                by_h[hb]["cov"] += 1
                best_i = max(overlapping, key=lambda t: t[1])[0]
                bw = boxes[best_i][2] - boxes[best_i][0]
                bh = max(1, boxes[best_i][3] - boxes[best_i][1])
                cover_box_ar.append(bw / bh)
                hit = [i for i, _ in overlapping if box_texts[i] == gt_text]
                if hit:
                    n_e2e_correct += 1
                    n_rec_correct_given_covered += 1
                    by_h[hb]["e2e"] += 1
                elif len(rec_fail_examples) < 20:
                    rec_fail_examples.append({
                        "img": img_name, "gt": gt_text, "best_box_pred": box_texts[best_i],
                        "n_overlap_boxes": len(overlapping),
                        "box_ar": round(bw / bh, 1), "gt_h": gt_h,
                    })
            else:
                best_c = max((_containment(b, gt_box) for b in boxes), default=0.0)
                if len(miss_examples) < 20:
                    miss_examples.append({"img": img_name, "gt": gt_text,
                                          "best_containment": round(best_c, 2), "gt_h": gt_h})

    cover_rate = n_covered / n_gt if n_gt else 0.0
    rec_em_cov = n_rec_correct_given_covered / n_covered if n_covered else 0.0
    e2e_em = n_e2e_correct / n_gt if n_gt else 0.0
    import statistics as st
    return {
        "crnn": str(crnn_pt), "det": str(det_model), "split": split,
        "det_long_side": det_long_side,
        "box_mode": box_mode,
        "unclip_ratio": unclip_ratio if box_mode == "quad" else None,
        "tail_model": str(tail_model) if tail_model else None,
        "tail_conf": tail_conf if tail_model else None,
        "resize_mode": resize_mode, "cover_thresh": cover_thresh,
        "n_images": n_images, "n_gt_testserial_regions": n_gt,
        "n_det_boxes": n_det_boxes,
        "detection_coverage": round(cover_rate * 100, 2),
        "rec_EM_given_covered": round(rec_em_cov * 100, 2),
        "e2e_EM": round(e2e_em * 100, 2),
        "cover_box_ar_median": round(st.median(cover_box_ar), 2) if cover_box_ar else None,
        "box_serial_fire": n_box_serial_fire,
        "box_false_fire": n_box_false_fire,
        "by_gt_height": {
            k: {"n": v["n"],
                "coverage": round(v["cov"] / v["n"] * 100, 1) if v["n"] else 0,
                "e2e_EM": round(v["e2e"] / v["n"] * 100, 1) if v["n"] else 0}
            for k, v in sorted(by_h.items())
        },
        "miss_examples": miss_examples,
        "rec_fail_examples": rec_fail_examples,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="end-to-end (paddle det -> CRNN) 実測")
    p.add_argument("--crnn", type=Path, required=True, help="CRNN .pt")
    p.add_argument("--det", type=Path, default=Path("dist-uranus2/model/paddle/ppocrv4_det.onnx"))
    p.add_argument("--samples-dir", type=Path, default=Path("samples"))
    p.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    p.add_argument("--labels", type=Path, default=Path("data/recognition_pad/labels_serial_split.tsv"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--cover-thresh", type=float, default=0.5, help="det box が GT を覆う containment 閾値")
    p.add_argument("--det-long-side", type=int, default=736, help="paddle det の長辺リサイズ(本番=736)")
    p.add_argument("--box-mode", choices=["rect", "quad"], default="rect",
                   help="rect=現行(CC→axis-aligned bbox) / quad=本家準拠(minAreaRect→回転矯正crop)")
    p.add_argument("--unclip-ratio", type=float, default=1.6, help="quad モードの unclip 倍率")
    p.add_argument("--tail-model", type=Path, default=None,
                   help="末尾2nd-pass 専用モデル (.pt)。quad モードでのみ有効")
    p.add_argument("--tail-conf", type=float, default=0.95)
    p.add_argument("--sim-app-preprocess", choices=["none", "gray2x", "binarize2x"],
                   default="none", help="URANUS2 の前処理を再現して実機入力分布で測る")
    p.add_argument("--limit", type=int, default=0, help="評価画像数上限 (0=全件)")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    res = evaluate(args.crnn, args.det, args.samples_dir, args.annotations_dir,
                   args.labels, args.split, args.cover_thresh, args.limit, args.det_long_side,
                   args.box_mode, args.unclip_ratio, args.tail_model, args.tail_conf,
                   args.sim_app_preprocess)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("miss_examples", "rec_fail_examples")},
                     ensure_ascii=False, indent=2))
    print("\nmiss(検出ミス)例:", json.dumps(res["miss_examples"][:8], ensure_ascii=False))
    print("rec誤読(覆ったが誤)例:", json.dumps(res["rec_fail_examples"][:8], ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
