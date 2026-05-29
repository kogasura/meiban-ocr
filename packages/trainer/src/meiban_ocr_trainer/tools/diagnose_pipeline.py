"""Detector vs Recognizer 切り分け診断 + 古典 CV pre-filter 評価 (Phase 2a)。

検出器 (sliding-window) と認識器 (CRNN+CTC) のうち、どちらが
「無駄な発火」「3秒の遅さ」の主因かを定量化する。さらに各窓の **古典 CV 特徴**
(エッジ密度・局所分散) を計算し、pre-filter でどこまで undef を削れるかを評価する。

手法:
  1. runtime と同じ sliding-window を Python で再現 (packages/runtime/src/detectors/sliding-window.ts)
  2. 同じ preprocess で各窓を 32×128 グレースケール正規化 (preprocess.ts)
  3. 各窓のエッジ密度 (Sobel) + 局所分散 (mean(x²) - mean(x)²) を全画像 1 度の畳み込みで計算
  4. 配信済 ONNX (models/meiban-ocr-v1.onnx, npm 0.3.2 同物) で推論 (オプションで pre-filter)
  5. annotation の positive/negative bbox と IoU で照合
  6. カテゴリ別 (pos/neg/undef) に特徴量分布・推論結果・pattern マッチ率・confidence を集計

出力カテゴリ:
  - **pos**: 窓が annotation の positive と IoU >= threshold (検出器が正しく取った)
  - **neg**: 窓が annotation の negative (= 人手で「非コード」認定済) と IoU >= threshold
  - **undef**: どちらにも該当しない (= 純背景・未アノテーション領域、おそらく noise)

Usage:
    # baseline (pre-filter 無し)
    python -m meiban_ocr_trainer.tools.diagnose_pipeline \\
        --onnx models/meiban-ocr-v1.onnx

    # pre-filter で頻度比較 (エッジ密度 >= 30 かつ 分散 >= 200 のみ ONNX に通す)
    python -m meiban_ocr_trainer.tools.diagnose_pipeline \\
        --edge-threshold 30 --var-threshold 200
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from meiban_ocr_trainer.constants import INPUT_HEIGHT, INPUT_WIDTH
from meiban_ocr_trainer.data.annotation import load_annotation
from meiban_ocr_trainer.tokenizer import CTCTokenizer
from meiban_ocr_trainer.vendors import ERICSSON

# ===== sliding-window (mirror of packages/runtime/src/detectors/sliding-window.ts) =====

SW_DEFAULTS = {
    "max_input_dim": 1024,
    "window_w": INPUT_WIDTH,   # 128
    "window_h": INPUT_HEIGHT,  # 32
    "stride_x": 32,
    "stride_y": 16,
    "scales": [1.0],
    "hard_limit": 20000,
}


def _compute_downscale(w: int, h: int, max_dim: int) -> float:
    long_side = max(w, h)
    return max_dim / long_side if long_side > max_dim else 1.0


def generate_windows(w: int, h: int, opts: dict | None = None) -> list[list[int]]:
    """runtime の generateWindowBoxes と同じ列挙ロジック (single scale 想定)。"""
    opts = {**SW_DEFAULTS, **(opts or {})}
    out: list[list[int]] = []
    ww, wh = opts["window_w"], opts["window_h"]
    sx, sy = opts["stride_x"], opts["stride_y"]
    for user_scale in opts["scales"]:
        base = _compute_downscale(w, h, opts["max_input_dim"])
        scale = base * user_scale
        det_w = round(w * scale)
        det_h = round(h * scale)
        if det_w < ww or det_h < wh:
            continue

        def to_image(x: int, y: int) -> list[int]:
            return [
                round(x / scale),
                round(y / scale),
                round((x + ww) / scale),
                round((y + wh) / scale),
            ]

        xs = list(range(0, det_w - ww + 1, sx))
        ys = list(range(0, det_h - wh + 1, sy))
        for y in ys:
            for x in xs:
                out.append(to_image(x, y))
        # edge windows (matches TS implementation)
        rem_x = (det_w - ww) % sx
        rem_y = (det_h - wh) % sy
        if rem_x > 0:
            x = det_w - ww
            for y in ys:
                out.append(to_image(x, y))
        if rem_y > 0:
            y = det_h - wh
            for x in xs:
                out.append(to_image(x, y))
        if len(out) > opts["hard_limit"]:
            raise RuntimeError(
                f"proposal count > {opts['hard_limit']} — stride too small"
            )
    return out


# ===== preprocess (mirror of packages/runtime/src/preprocess.ts) =====

def crop_and_normalize(img_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    """bbox 領域を 32×128 にリサイズ → Rec.709 グレースケール → [-1, 1] 正規化。"""
    x1, y1, x2, y2 = bbox
    h, w = img_rgb.shape[:2]
    # bbox を画像境界に clip (runtime 側は canvas が自動で fill する想定)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    crop = img_rgb[y1c:y2c, x1c:x2c]
    # cv2.resize は (W, H) 順
    resized = cv2.resize(crop, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    # Rec.709 luminance
    y = (0.2126 * resized[..., 0]
         + 0.7152 * resized[..., 1]
         + 0.0722 * resized[..., 2])
    # mean=0.5, std=0.5
    return ((y / 255.0 - 0.5) / 0.5).astype(np.float32)


# ===== 古典 CV pre-filter 特徴 (Phase 2a) =====

def compute_edge_map(img_gray: np.ndarray) -> np.ndarray:
    """Sobel ベースのエッジ強度マップ (各画素の |∇I|)。

    1 画像 1 回だけ計算し、窓ごとに mean を取れば窓特徴になる。
    cv2 の Sobel は SIMD 最適化されており、1280×960 で ~3 ms。
    """
    sx = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx * sx + sy * sy)


def compute_local_var_map(img_gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """ksize × ksize 局所分散マップ: var = E[x²] - E[x]²。"""
    gf = img_gray.astype(np.float32)
    mean = cv2.boxFilter(gf, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    sq = cv2.boxFilter(gf * gf, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    var = sq - mean * mean
    return np.maximum(var, 0.0)


def window_features(
    edge_map: np.ndarray,
    var_map: np.ndarray,
    bbox: list[int],
) -> tuple[float, float]:
    """窓のエッジ密度・局所分散の平均を返す。bbox は画像座標。"""
    x1, y1, x2, y2 = bbox
    h, w = edge_map.shape
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return (0.0, 0.0)
    edge_mean = float(edge_map[y1c:y2c, x1c:x2c].mean())
    var_mean = float(var_map[y1c:y2c, x1c:x2c].mean())
    return (edge_mean, var_mean)


def _percentile_summary(values: list[float]) -> dict[str, float]:
    """min/p25/p50/p75/max + mean を辞書で返す。"""
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float32)
    return {
        "n": len(values),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


# ===== IoU =====

def iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    b_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def classify_bbox(
    bbox: list[int],
    pos_bboxes: list[list[int]],
    neg_bboxes: list[list[int]],
    threshold: float,
) -> tuple[str, float]:
    """bbox がどのカテゴリに属するか + max IoU。"""
    best_pos = max((iou(bbox, p) for p in pos_bboxes), default=0.0)
    best_neg = max((iou(bbox, n) for n in neg_bboxes), default=0.0)
    if best_pos >= threshold and best_pos >= best_neg:
        return ("pos", best_pos)
    if best_neg >= threshold:
        return ("neg", best_neg)
    return ("undef", max(best_pos, best_neg))


def containment_ratio(window: list[int], target: list[int]) -> float:
    """target (text_bbox) が window にどれだけ含まれているか = intersection / target_area。

    認識器の入力として「文字が窓に収まっているか」を測る指標。
    IoU と違い、window が target より大きい場合でも 1.0 になりうる (= 文字が完全に窓内)。
    """
    wx1, wy1, wx2, wy2 = window
    tx1, ty1, tx2, ty2 = target
    ix1, iy1 = max(wx1, tx1), max(wy1, ty1)
    ix2, iy2 = min(wx2, tx2), min(wy2, ty2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    target_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
    return inter / target_area if target_area > 0 else 0.0


def evaluate_detector(
    windows: list[list[int]],
    pos_bboxes: list[list[int]],
    pred_per_window: list[tuple[str, float, bool]],
) -> dict:
    """検出器の本来の指標を計算する。

    各 GT positive について:
        - best_iou: max IoU across all windows
        - best_containment: max (intersection / text_bbox_area) — 認識器が文字全体を見られるか
        - covered@k: IoU >= k の窓が 1 つ以上あるか
        - recognized: その positive を覆う窓のうち、pat-match + conf>=threshold が 1 つ以上あるか

    pred_per_window: 各窓の (pred_text, conf, is_pattern_high_conf)
    """
    per_pos = []
    for gt in pos_bboxes:
        best_iou = 0.0
        best_containment = 0.0
        recognized = False
        for w, (pred, conf, is_high) in zip(windows, pred_per_window):
            cur_iou = iou(w, gt)
            cur_cont = containment_ratio(w, gt)
            if cur_iou > best_iou:
                best_iou = cur_iou
            if cur_cont > best_containment:
                best_containment = cur_cont
            if cur_iou >= 0.3 and is_high:
                recognized = True
        per_pos.append({
            "gt": gt,
            "best_iou": best_iou,
            "best_containment": best_containment,
            "covered_03": best_iou >= 0.3,
            "covered_05": best_iou >= 0.5,
            "covered_07": best_iou >= 0.7,
            "fully_contained_09": best_containment >= 0.9,
            "recognized": recognized,
        })

    n = len(per_pos)
    return {
        "n_positives": n,
        "coverage_03": sum(p["covered_03"] for p in per_pos) / max(n, 1),
        "coverage_05": sum(p["covered_05"] for p in per_pos) / max(n, 1),
        "coverage_07": sum(p["covered_07"] for p in per_pos) / max(n, 1),
        "fully_contained_09": sum(p["fully_contained_09"] for p in per_pos) / max(n, 1),
        "end_to_end_recall": sum(p["recognized"] for p in per_pos) / max(n, 1),
        "iou_summary": _percentile_summary([p["best_iou"] for p in per_pos]),
        "containment_summary": _percentile_summary(
            [p["best_containment"] for p in per_pos]
        ),
        "per_pos": per_pos,
    }


# ===== diagnostic main =====

def diagnose_image(
    image_path: Path,
    ann_path: Path,
    session: ort.InferenceSession,
    sw_opts: dict,
    iou_threshold: float,
    conf_threshold: float,
    edge_threshold: float | None = None,
    var_threshold: float | None = None,
    batch_size: int = 64,
) -> dict:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"failed to read {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]

    # 1. sliding-window
    t0 = time.time()
    windows = generate_windows(w, h, sw_opts)
    t_windows = time.time() - t0

    # 2. pre-filter 特徴量 (画像全体に対し 1 度だけ)
    t1 = time.time()
    edge_map = compute_edge_map(img_gray)
    var_map = compute_local_var_map(img_gray, ksize=7)
    t_feature_maps = time.time() - t1

    t2 = time.time()
    window_feats = [window_features(edge_map, var_map, b) for b in windows]
    t_window_feats = time.time() - t2

    # pre-filter で除外する窓を決定
    passes_filter: list[bool] = []
    for edge, var in window_feats:
        ok = True
        if edge_threshold is not None and edge < edge_threshold:
            ok = False
        if var_threshold is not None and var < var_threshold:
            ok = False
        passes_filter.append(ok)
    n_passed = sum(passes_filter)
    pass_rate = n_passed / max(len(windows), 1)

    print(
        f"  windows: {len(windows)} (sw {t_windows*1000:.1f}ms + "
        f"feat-maps {t_feature_maps*1000:.1f}ms + "
        f"feat-extract {t_window_feats*1000:.1f}ms)",
        file=sys.stderr,
    )
    if edge_threshold is not None or var_threshold is not None:
        print(
            f"  pre-filter (edge>={edge_threshold}, var>={var_threshold}): "
            f"{n_passed}/{len(windows)} passed ({100*pass_rate:.1f}%)",
            file=sys.stderr,
        )

    # 3. annotation classification
    ann = load_annotation(ann_path)
    pos_bboxes = [list(r.text_bbox or r.bbox) for r in ann.positives]
    neg_bboxes = [list(r.bbox) for r in ann.negatives]
    classifications = [
        classify_bbox(b, pos_bboxes, neg_bboxes, iou_threshold) for b in windows
    ]

    # 4. ONNX 推論 (バッチ、pre-filter 通過のみ)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    tokenizer = CTCTokenizer()
    pattern = ERICSSON.strict_regex

    preds: list[str] = [""] * len(windows)
    confs: list[float] = [0.0] * len(windows)
    inference_indices = [i for i, p in enumerate(passes_filter) if p]
    t_infer = 0.0
    for i in range(0, len(inference_indices), batch_size):
        batch_idx = inference_indices[i:i + batch_size]
        batch = [windows[j] for j in batch_idx]
        crops = np.stack([crop_and_normalize(img_rgb, b) for b in batch])
        x = crops[:, None, :, :]  # (N, 1, H, W)
        t3 = time.time()
        logits_np = session.run([output_name], {input_name: x})[0]
        t_infer += time.time() - t3
        logits_t = torch.from_numpy(logits_np)
        for j, (text, conf) in zip(batch_idx,
                                   tokenizer.greedy_decode_with_conf(logits_t)):
            preds[j] = text
            confs[j] = conf

    print(
        f"  inference {n_passed} windows in {t_infer:.3f}s "
        f"({1000 * t_infer / max(n_passed, 1):.2f} ms/window)",
        file=sys.stderr,
    )

    # 5. カテゴリ別集計 (特徴量分布も入れる)
    stats: dict[str, dict] = {}
    for cat in ("pos", "neg", "undef"):
        stats[cat] = {
            "n": 0, "n_passed_filter": 0,
            "pat": 0, "pat_high": 0, "nonempty": 0, "conf_sum": 0.0,
            "samples": [],
            "edges": [], "vars": [],
        }
    pred_per_window: list[tuple[str, float, bool]] = []
    for (cat, _), pred, conf, passes, (edge, var) in zip(
        classifications, preds, confs, passes_filter, window_feats,
    ):
        s = stats[cat]
        s["n"] += 1
        s["edges"].append(edge)
        s["vars"].append(var)
        if passes:
            s["n_passed_filter"] += 1
        s["conf_sum"] += conf
        if pred:
            s["nonempty"] += 1
        is_pat_high = bool(pattern.match(pred)) and conf >= conf_threshold
        pred_per_window.append((pred, conf, is_pat_high))
        if pattern.match(pred):
            s["pat"] += 1
            if conf >= conf_threshold:
                s["pat_high"] += 1
            if len(s["samples"]) < 5 or conf > min(x["conf"] for x in s["samples"]):
                s["samples"].append({"pred": pred, "conf": conf, "edge": edge,
                                     "var": var})
                s["samples"].sort(key=lambda x: -x["conf"])
                s["samples"] = s["samples"][:5]

    # 特徴量を percentile に集約
    for cat in ("pos", "neg", "undef"):
        s = stats[cat]
        s["edge_summary"] = _percentile_summary(s.pop("edges"))
        s["var_summary"] = _percentile_summary(s.pop("vars"))

    # 6. 検出器の本来評価 (per-GT-positive coverage + end-to-end recall)
    detector_eval = evaluate_detector(windows, pos_bboxes, pred_per_window)

    return {
        "image": image_path.name,
        "image_size": [w, h],
        "n_windows": len(windows),
        "n_pos_ann": len(pos_bboxes),
        "n_neg_ann": len(neg_bboxes),
        "t_windows_sec": t_windows,
        "t_feature_maps_sec": t_feature_maps,
        "t_window_feats_sec": t_window_feats,
        "t_infer_sec": t_infer,
        "n_passed_filter": n_passed,
        "stats": stats,
        "detector_eval": detector_eval,
    }


def _format_report(result: dict, conf_threshold: float) -> str:
    img = result["image"]
    n_total = result["n_windows"]
    de = result["detector_eval"]
    iou_s = de["iou_summary"]
    cont_s = de["containment_summary"]
    lines = [
        f"\n=== {img} ({result['image_size'][0]}×{result['image_size'][1]}, "
        f"ann: pos={result['n_pos_ann']} / neg={result['n_neg_ann']}) ===",
        f"  total windows: {n_total}",
        f"  inference time: {result['t_infer_sec']:.2f}s "
        f"({1000 * result['t_infer_sec'] / max(n_total, 1):.1f} ms/window)",
        "",
        "  --- 検出器評価 (GT positive 単位) ---",
        f"  GT positives: {de['n_positives']}",
        f"  Coverage@IoU≥0.3: {100*de['coverage_03']:>5.1f}%   "
        f"@IoU≥0.5: {100*de['coverage_05']:>5.1f}%   "
        f"@IoU≥0.7: {100*de['coverage_07']:>5.1f}%",
        f"  text_bbox を 90%以上含む窓あり: {100*de['fully_contained_09']:>5.1f}%",
        f"  best-IoU per GT  p25/p50/p75: "
        f"{iou_s.get('p25',0):.2f} / {iou_s.get('p50',0):.2f} / {iou_s.get('p75',0):.2f}  "
        f"(min={iou_s.get('min',0):.2f}, max={iou_s.get('max',0):.2f})",
        f"  containment per GT p25/p50/p75: "
        f"{cont_s.get('p25',0):.2f} / {cont_s.get('p50',0):.2f} / "
        f"{cont_s.get('p75',0):.2f}",
        f"  end-to-end recall (IoU≥0.3 かつ pat+conf≥{conf_threshold}): "
        f"{100*de['end_to_end_recall']:>5.1f}%",
        "",
        "  --- 窓カテゴリ分布 ---",
        f"  {'category':<8} {'count':>6} {'%':>6}   "
        f"{'pattern_match':>14} ({'≥' + f'{conf_threshold:.1f}':>4})  "
        f"{'nonempty':>9} {'avg_conf':>9}",
    ]
    for cat in ("pos", "neg", "undef"):
        s = result["stats"][cat]
        n = s["n"]
        if n == 0:
            continue
        pct = 100 * n / max(n_total, 1)
        pat = s["pat"]
        pat_high = s["pat_high"]
        pat_pct = 100 * pat / max(n, 1)
        high_pct = 100 * pat_high / max(n, 1)
        nonempty = s["nonempty"]
        nonempty_pct = 100 * nonempty / max(n, 1)
        avg_conf = s["conf_sum"] / max(n, 1)
        n_passed = s["n_passed_filter"]
        pass_pct = 100 * n_passed / max(n, 1)
        lines.append(
            f"  {cat:<8} {n:>6} {pct:>5.1f}%   "
            f"{pat:>6} ({pat_pct:>5.1f}%)  ({pat_high:>5} {high_pct:>5.1f}%)  "
            f"{nonempty_pct:>7.1f}% {avg_conf:>9.3f}    pass_filter={pass_pct:>5.1f}%"
        )
        # 特徴量分布
        es = s.get("edge_summary", {})
        vs = s.get("var_summary", {})
        if es.get("n", 0):
            lines.append(
                f"    edge   p25/p50/p75: "
                f"{es['p25']:>6.1f} / {es['p50']:>6.1f} / {es['p75']:>6.1f}"
            )
        if vs.get("n", 0):
            lines.append(
                f"    var    p25/p50/p75: "
                f"{vs['p25']:>6.1f} / {vs['p50']:>6.1f} / {vs['p75']:>6.1f}"
            )
        # sample preds
        if s["samples"]:
            samples_str = ", ".join(
                f"{x['pred']!r}@{x['conf']:.2f}(e={x['edge']:.0f},v={x['var']:.0f})"
                for x in s["samples"][:3]
            )
            lines.append(f"    ↳ top pattern-match preds: {samples_str}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose detector vs recognizer.")
    parser.add_argument("--onnx", type=Path,
                        default=Path("models/meiban-ocr-v1.onnx"))
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="bbox を pos/neg に分類するための IoU 閾値 (default 0.3)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="高 confidence と判定する閾値 (default 0.7)",
    )
    parser.add_argument(
        "--scales", type=str, default="1.0",
        help="sliding-window のスケール、カンマ区切り (例: '0.7,1.0,1.4')",
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=None,
        help="Sobel エッジ平均がこの値未満なら pre-filter で skip (default: 無効)",
    )
    parser.add_argument(
        "--var-threshold", type=float, default=None,
        help="局所分散平均がこの値未満なら pre-filter で skip (default: 無効)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    scales = [float(s.strip()) for s in args.scales.split(",")]
    sw_opts = {"scales": scales}

    if not args.onnx.exists():
        print(f"[diagnose] onnx not found: {args.onnx}", file=sys.stderr)
        return 1

    print(f"[diagnose] loading ONNX: {args.onnx}", file=sys.stderr)
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"],
    )

    images = sorted(args.samples_dir.glob("img_*.jpg"))
    if not images:
        print(f"[diagnose] no images in {args.samples_dir}", file=sys.stderr)
        return 1

    results = []
    for img_path in images:
        ann_path = args.annotations_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            print(f"  - {img_path.name}: no annotation, skip", file=sys.stderr)
            continue
        print(f"\n--- processing {img_path.name} ---", file=sys.stderr)
        r = diagnose_image(
            img_path, ann_path, session,
            sw_opts=sw_opts,
            iou_threshold=args.iou_threshold,
            conf_threshold=args.confidence_threshold,
            edge_threshold=args.edge_threshold,
            var_threshold=args.var_threshold,
            batch_size=args.batch_size,
        )
        print(_format_report(r, args.confidence_threshold))
        results.append(r)

    # 全画像合算
    print("\n\n========== AGGREGATE ==========")
    agg = {c: {"n": 0, "pat": 0, "pat_high": 0, "nonempty": 0, "conf_sum": 0.0}
           for c in ("pos", "neg", "undef")}
    total_windows = 0
    total_t_infer = 0.0
    total_pos_gt = 0
    total_cov_03 = 0
    total_cov_05 = 0
    total_cov_07 = 0
    total_fc_09 = 0
    total_e2e = 0
    all_iou = []
    all_cont = []
    for r in results:
        total_windows += r["n_windows"]
        total_t_infer += r["t_infer_sec"]
        for c in ("pos", "neg", "undef"):
            for k in ("n", "pat", "pat_high", "nonempty"):
                agg[c][k] += r["stats"][c][k]
            agg[c]["conf_sum"] += r["stats"][c]["conf_sum"]
        de = r["detector_eval"]
        n_pos = de["n_positives"]
        total_pos_gt += n_pos
        total_cov_03 += int(round(de["coverage_03"] * n_pos))
        total_cov_05 += int(round(de["coverage_05"] * n_pos))
        total_cov_07 += int(round(de["coverage_07"] * n_pos))
        total_fc_09 += int(round(de["fully_contained_09"] * n_pos))
        total_e2e += int(round(de["end_to_end_recall"] * n_pos))
        all_iou.extend(p["best_iou"] for p in de["per_pos"])
        all_cont.extend(p["best_containment"] for p in de["per_pos"])

    print(f"  {len(results)} images, {total_windows} total windows")
    print(f"  total inference: {total_t_infer:.2f}s "
          f"({1000 * total_t_infer / max(total_windows, 1):.1f} ms/window)")
    print("\n  --- 検出器 (sliding-window) 評価 ---")
    print(f"  GT positives total: {total_pos_gt}")
    print(f"  Coverage@IoU≥0.3: {100 * total_cov_03 / max(total_pos_gt, 1):.1f}% "
          f"({total_cov_03}/{total_pos_gt})")
    print(f"  Coverage@IoU≥0.5: {100 * total_cov_05 / max(total_pos_gt, 1):.1f}% "
          f"({total_cov_05}/{total_pos_gt})")
    print(f"  Coverage@IoU≥0.7: {100 * total_cov_07 / max(total_pos_gt, 1):.1f}% "
          f"({total_cov_07}/{total_pos_gt})")
    print(f"  text_bbox 90%以上含む: {100 * total_fc_09 / max(total_pos_gt, 1):.1f}%")
    print(f"  end-to-end recall (IoU≥0.3 + pat+conf≥{args.confidence_threshold}): "
          f"{100 * total_e2e / max(total_pos_gt, 1):.1f}% ({total_e2e}/{total_pos_gt})")
    iou_s = _percentile_summary(all_iou)
    cont_s = _percentile_summary(all_cont)
    print(f"  best-IoU p25/p50/p75: "
          f"{iou_s.get('p25',0):.2f} / {iou_s.get('p50',0):.2f} / "
          f"{iou_s.get('p75',0):.2f}")
    print(f"  containment p25/p50/p75: "
          f"{cont_s.get('p25',0):.2f} / {cont_s.get('p50',0):.2f} / "
          f"{cont_s.get('p75',0):.2f}")

    print(
        f"\n  --- 窓カテゴリ分布 ---\n"
        f"  {'category':<8} {'count':>6} {'%':>6}   "
        f"{'pattern_match':>14}  (≥{args.confidence_threshold:.1f}%)  "
        f"{'nonempty':>9} {'avg_conf':>9}"
    )
    for c in ("pos", "neg", "undef"):
        n = agg[c]["n"]
        if n == 0:
            continue
        pct = 100 * n / max(total_windows, 1)
        pat = agg[c]["pat"]
        pat_high = agg[c]["pat_high"]
        nonempty_pct = 100 * agg[c]["nonempty"] / max(n, 1)
        avg_conf = agg[c]["conf_sum"] / max(n, 1)
        print(
            f"  {c:<8} {n:>6} {pct:>5.1f}%   "
            f"{pat:>6} ({100 * pat / max(n, 1):>5.1f}%)  "
            f"({pat_high:>5} {100 * pat_high / max(n, 1):>5.1f}%)  "
            f"{nonempty_pct:>7.1f}% {avg_conf:>9.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
