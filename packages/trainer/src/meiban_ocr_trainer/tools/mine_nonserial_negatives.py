"""非シリアル negative を paddle det で採掘して annotation に追記する (reject 訓練用)。

ハイブリッド (paddle det 検出 → custom rec 認識) の誤発火根因は、custom 認識器に reject
機構が無く、paddle det が拾う「非シリアルのテキスト」(型番 / 製造番号ラベル / 製造年月 /
会社名 / 反射 / グレア) に対しても Ericsson 形式のシリアルを捏造することにある。
既存の negative annotation は全て subkind=background (無文字) で、この「非シリアルの文字」が
学習されていない。

本スクリプトは PP-OCRv4 det を samples に回し、各画像の **positive(シリアル) 領域と重ならない
検出 box** を subkind="mined" の negative region として annotation に追記する。これにより
extract_crops が ∅ ラベルの非シリアル crop を生成し、12-head reject 訓練の負例になる。

前処理/後処理は runtime TS (`packages/runtime/src/backends/paddle/preprocess.ts:preprocessForDet`,
`backends/paddle/db_postprocess.ts:dbPostprocess`) を numpy へ移植したもの。runtime と同じ
box が出るように det の挙動を一致させている。

Usage:
    python -m meiban_ocr_trainer.tools.mine_nonserial_negatives \\
        --samples-dir samples --annotations-dir annotations \\
        --det-model ../dist-uranus2/model/paddle/ppocrv4_det.onnx \\
        --limit 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# --- paddle det 既定値 (runtime と一致させる) ---
DET_LIMIT_SIDE = 736          # preprocess.ts DET_DEFAULT_LIMIT
BINARY_THRESHOLD = 0.3        # db_postprocess.ts DEFAULTS.binaryThreshold
SCORE_THRESHOLD = 0.5         # DEFAULTS.scoreThreshold
MIN_BOX_SIZE = 3              # DEFAULTS.minBoxSize (seg 解像度上)
UNCLIP_RATIO = 1.6            # DEFAULTS.unclipRatio


def preprocess_for_det(img_rgb: np.ndarray, limit_side: int = DET_LIMIT_SIDE):
    """preprocessForDet 移植: 長辺<=limit, 32倍数, (v-127.5)/127.5, RGB CHW [1,3,H,W]."""
    orig_h, orig_w = img_rgb.shape[:2]
    long_side = max(orig_w, orig_h)
    scale = limit_side / long_side if long_side > limit_side else 1.0
    new_h = max(32, round(orig_h * scale / 32) * 32)
    new_w = max(32, round(orig_w * scale / 32) * 32)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) - 127.5) / 127.5
    chw = np.transpose(norm, (2, 0, 1))[None, ...]  # [1,3,H,W]
    return np.ascontiguousarray(chw, dtype=np.float32)


def db_postprocess(seg_map: np.ndarray, orig_h: int, orig_w: int):
    """dbPostprocess 移植: seg map -> 元画像座標の axis-aligned bbox [x1,y1,x2,y2]."""
    seg_h, seg_w = seg_map.shape
    binary = (seg_map >= BINARY_THRESHOLD).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return []

    # 各ラベルの seg 平均 score を効率的に算出
    flat_labels = labels.ravel()
    score_sum = np.bincount(flat_labels, weights=seg_map.ravel(), minlength=num)
    area = np.bincount(flat_labels, minlength=num)

    scale_x = orig_w / seg_w
    scale_y = orig_h / seg_h
    boxes: list[list[int]] = []
    for l in range(1, num):
        cnt = area[l]
        if cnt <= 0:
            continue
        score = score_sum[l] / cnt
        if score < SCORE_THRESHOLD:
            continue
        x, y, w, h = stats[l, 0], stats[l, 1], stats[l, 2], stats[l, 3]
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue
        # unclip: 中心 outward に ratio 倍 expand (TS と同じ半開区間 center)
        cx = x + w / 2.0
        cy = y + h / 2.0
        half_w = (w * UNCLIP_RATIO) / 2.0
        half_h = (h * UNCLIP_RATIO) / 2.0
        x1 = max(0, round((cx - half_w) * scale_x))
        y1 = max(0, round((cy - half_h) * scale_y))
        x2 = min(orig_w, round((cx + half_w) * scale_x))
        y2 = min(orig_h, round((cy + half_h) * scale_y))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _contains_center(box, inner) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _positive_boxes(regions) -> list[list[int]]:
    """positive(シリアル)領域の bbox 一覧。text_bbox 優先。"""
    out = []
    for r in regions:
        if r.get("category", "positive") != "positive":
            continue
        bb = r.get("text_bbox") or r.get("bbox")
        if bb:
            out.append(bb)
    return out


def mine_image(
    sess: ort.InferenceSession,
    in_name: str,
    out_name: str,
    img_path: Path,
    pos_boxes: list[list[int]],
    overlap_iou: float = 0.10,
) -> list[list[int]]:
    """1画像で paddle det を回し、positive と重ならない非シリアル box を返す。"""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]
    inp = preprocess_for_det(rgb)
    seg = sess.run([out_name], {in_name: inp})[0]  # [1,1,segH,segW]
    seg_map = np.asarray(seg)[0, 0].astype(np.float32)
    det_boxes = db_postprocess(seg_map, orig_h, orig_w)

    mined = []
    for box in det_boxes:
        # positive(シリアル)と重なる box は除外（シリアルを∅負例にしない安全側）
        overlaps_pos = any(
            _iou(box, pb) > overlap_iou or _contains_center(box, pb)
            for pb in pos_boxes
        )
        if overlaps_pos:
            continue
        mined.append(box)
    return mined


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mine non-serial negatives via paddle det.")
    p.add_argument("--samples-dir", type=Path, default=Path("samples"))
    p.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    p.add_argument(
        "--det-model", type=Path,
        default=Path("dist-uranus2/model/paddle/ppocrv4_det.onnx"),
    )
    p.add_argument("--limit", type=int, default=0, help="処理画像数の上限 (0=全件)")
    p.add_argument("--overlap-iou", type=float, default=0.10)
    p.add_argument("--max-per-image", type=int, default=12,
                   help="1画像あたりの mined negative 上限 (面積上位)")
    p.add_argument("--force", action="store_true",
                   help="既存の subkind=mined を再生成 (default は冪等skip)")
    args = p.parse_args(argv)

    if not args.det_model.exists():
        print(f"det model not found: {args.det_model}")
        return 1

    # GPU機では CUDA、無ければ CPU に自動フォールバック（onnxruntime が解決）。
    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(str(args.det_model), providers=providers)
    print(f"[mine] providers: {sess.get_providers()}")
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    ann_paths = sorted(args.annotations_dir.glob("img_*.json"))
    if args.limit > 0:
        ann_paths = ann_paths[: args.limit]

    total_mined = 0
    n_files = 0
    for idx, ann_path in enumerate(ann_paths):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        regions = data.get("regions", [])

        has_mined = any(r.get("subkind") == "mined" for r in regions)
        if has_mined and not args.force:
            continue
        if has_mined and args.force:
            regions = [r for r in regions if r.get("subkind") != "mined"]

        img_path = args.samples_dir / data.get("image", ann_path.stem + ".jpg")
        if not img_path.exists():
            continue

        pos_boxes = _positive_boxes(regions)
        mined = mine_image(sess, in_name, out_name, img_path, pos_boxes,
                           overlap_iou=args.overlap_iou)
        if not mined:
            continue
        # 面積上位 max_per_image に制限（過剰な小boxを抑える）
        mined.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        mined = mined[: args.max_per_image]

        next_id = max((r.get("id", -1) for r in regions), default=-1) + 1
        for box in mined:
            regions.append({
                "id": next_id,
                "category": "negative",
                "bbox": box,
                "subkind": "mined",
                "claude_verified": True,  # auto. positive非重複で安全側に採掘
            })
            next_id += 1
            total_mined += 1
        n_files += 1

        data["regions"] = regions
        ann_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        if (idx + 1) % 200 == 0:
            print(f"  ... {idx + 1}/{len(ann_paths)} 画像処理, mined={total_mined}")

    print(f"mined {total_mined} non-serial negatives across {n_files} images "
          f"(processed {len(ann_paths)} annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
