"""Session-level OCR evaluation: 動画 → custom model vs PaddleOCR (= GT) 比較。

KGI 評価のメインツール。 1 セッション (= 1 動画の最初の 3 秒、 default) で:

  1. PaddleOCR (RapidOCR) が動画全体から検出した unique Ericsson serial を **GT 集合**
  2. Custom ONNX で sliding-window + recognizer 推論、 frame ごとに accept された serial 収集
  3. Optional: k-of-n consensus (同 text が k frame 以上で出たら確定)
  4. Metrics:
     - recall = |custom ∩ truth| / |truth|     (KGI: 100%)
     - precision = |custom ∩ truth| / |custom| (補助)
     - false_positives = |custom - truth|       (KGI: 0)
     - misses = |truth - custom|                (KGI: 0)
     - mean_frame_latency_ms (custom model, KGI ≤ 300ms)
     - first_complete_sec (= 全 GT を初めて読み終えた時間、 KGI ≤ 3s)

Usage:
    python -m meiban_ocr_trainer.tools.session_diagnose \\
        --video videos/10_44.mov \\
        --onnx models/meiban-ocr-v2-fh.onnx \\
        --fps 3.3 \\
        --duration 3.0 \\
        --custom-conf 0.7 \\
        --consensus-k 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from meiban_ocr_trainer.constants import INPUT_HEIGHT, INPUT_WIDTH
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    SW_DEFAULTS,
    _decode_logits,
    _detect_model_type_and_tokenizer,
    crop_and_normalize,
    generate_windows,
)
from meiban_ocr_trainer.vendors import ERICSSON

ERICSSON_STRICT = ERICSSON.strict_regex
ERICSSON_PARTIAL = ERICSSON.partial_regex


def _normalize(text: str) -> str:
    """auto_label と同じ正規化 (NFKC + uppercase + dash 除去)。"""
    t = unicodedata.normalize("NFKC", text or "")
    return t.upper().replace("-", "").replace(" ", "")


def _extract_ericsson_from_rapidocr(raw_entries) -> list[tuple[str, float]]:
    """RapidOCR 出力から Ericsson pattern にマッチする (text, conf) を取り出す。"""
    out: list[tuple[str, float]] = []
    for entry in raw_entries or []:
        _, raw_text, conf = entry[0], entry[1], float(entry[2])
        text = _normalize(raw_text)
        if ERICSSON_STRICT.match(text):
            out.append((text, conf))
        else:
            m = ERICSSON_PARTIAL.search(text)
            if m:
                out.append((m.group(0), conf))
    return out


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_sec: float
    custom_accepted: list[tuple[str, float]] = field(default_factory=list)  # (text, conf)
    gt_serials: list[tuple[str, float]] = field(default_factory=list)
    inference_ms: float = 0.0


@dataclass
class SessionMetrics:
    truth: set[str]
    custom: set[str]
    intersect: set[str]
    false_positives: set[str]
    misses: set[str]
    recall: float
    precision: float
    mean_frame_latency_ms: float
    p95_frame_latency_ms: float
    first_complete_sec: float | None
    n_frames: int
    n_session_frames: int
    # 2026-06-02 追加: KGI Score (誤検出を大きめに減点する複合スコア)。
    # = recall × 100 − FP_count × fp_penalty、 範囲 [0, 100]。
    kgi_score: float = 0.0
    kgi_fp_penalty: float = 25.0


def _sample_video_frames(
    video_path: Path,
    fps_sample: float,
    max_duration_sec: float | None = None,
) -> list[tuple[int, float, np.ndarray]]:
    """動画から fps_sample で間引いた frame を (frame_idx, timestamp_sec, bgr) で返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(round(src_fps / max(0.01, fps_sample))))
        frames: list[tuple[int, float, np.ndarray]] = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                ts = idx / src_fps
                if max_duration_sec is not None and ts > max_duration_sec:
                    break
                frames.append((idx, ts, bgr))
            idx += 1
        return frames
    finally:
        cap.release()


def _run_custom_model_on_frame(
    img_bgr: np.ndarray,
    session: ort.InferenceSession,
    model_type: str,
    tokenizer,
    sw_opts: dict,
    conf_threshold: float,
    batch_size: int,
) -> tuple[list[tuple[str, float]], float]:
    """custom model で 1 frame の sliding-window 推論 → accept された (text, conf) と推論時間 ms。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    windows = generate_windows(w, h, sw_opts)

    accepted: list[tuple[str, float]] = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    t0 = time.time()
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        crops = np.stack([crop_and_normalize(img_rgb, b) for b in batch])
        x = crops[:, None, :, :]
        logits_np = session.run([output_name], {input_name: x})[0]
        decoded = _decode_logits(model_type, tokenizer, logits_np)
        for (text, conf) in decoded:
            if conf < conf_threshold:
                continue
            if not text:
                continue
            if not ERICSSON_STRICT.match(text):
                # partial も拾う (= 部分一致を許容、 max conf で aggregate)
                m = ERICSSON_PARTIAL.search(text)
                if not m:
                    continue
                text = m.group(0)
            accepted.append((text, conf))
    elapsed_ms = (time.time() - t0) * 1000.0
    return accepted, elapsed_ms


def _aggregate_consensus(
    per_frame: list[list[tuple[str, float]]],
    k: int,
) -> dict[str, float]:
    """k-of-n consensus: 各 text が出現した frame 数 ≥ k なら accept。

    Returns: {text: max_conf} の dict。
    """
    text_frames: dict[str, int] = {}
    text_max_conf: dict[str, float] = {}
    for frame_results in per_frame:
        seen_this_frame: set[str] = set()
        for text, conf in frame_results:
            if text in seen_this_frame:
                continue
            seen_this_frame.add(text)
            text_frames[text] = text_frames.get(text, 0) + 1
            if conf > text_max_conf.get(text, 0.0):
                text_max_conf[text] = conf
    return {t: c for t, c in text_max_conf.items() if text_frames.get(t, 0) >= k}


def evaluate_session(
    video_path: Path,
    onnx_path: Path,
    fps_sample: float = 3.3,
    duration_sec: float = 3.0,
    custom_conf_threshold: float = 0.7,
    paddle_conf_threshold: float = 0.9,
    consensus_k: int = 3,
    batch_size: int = 32,
    sw_opts: dict | None = None,
    fp_penalty: float = 25.0,
) -> tuple[SessionMetrics, list[FrameResult]]:
    """1 動画に対するセッション評価。"""
    sw_opts = {**SW_DEFAULTS, **(sw_opts or {})}

    print(f"[session_diagnose] video: {video_path}", file=sys.stderr)
    print(f"  fps_sample={fps_sample}, duration={duration_sec}s, k={consensus_k}",
          file=sys.stderr)

    # 1. Custom ONNX session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_type, tokenizer = _detect_model_type_and_tokenizer(session)
    print(f"  model_type={model_type}", file=sys.stderr)

    # 2. PaddleOCR (RapidOCR) for GT
    from rapidocr_onnxruntime import RapidOCR
    rapid = RapidOCR()

    # 3. Frame sampling — GT は動画全体、 セッション評価は最初の duration_sec
    print("  sampling frames (full video for GT)...", file=sys.stderr)
    all_frames = _sample_video_frames(video_path, fps_sample, max_duration_sec=None)
    print(f"  sampled {len(all_frames)} frames", file=sys.stderr)

    # 4. GT 抽出: 全 frame で RapidOCR、 unique Ericsson 集合
    print("  GT extraction via PaddleOCR...", file=sys.stderr)
    gt_set: set[str] = set()
    gt_per_frame: list[list[tuple[str, float]]] = []
    for (fi, ts, bgr) in all_frames:
        # RapidOCR が numpy 直接受け取れない場合は temp file 経由
        # RapidOCR は __call__ で numpy も受け取れる
        try:
            result, _ = rapid(bgr)
        except Exception as e:  # 画像形式エラー等のフォールバック
            tmp = Path(f"/tmp/_session_frame_{fi}.jpg")
            cv2.imwrite(str(tmp), bgr)
            try:
                result, _ = rapid(str(tmp))
            finally:
                if tmp.exists():
                    tmp.unlink()
        ericssons = _extract_ericsson_from_rapidocr(result)
        # GT は paddle_conf_threshold 以上のみ採用
        hi_conf = [(t, c) for t, c in ericssons if c >= paddle_conf_threshold]
        for t, _ in hi_conf:
            gt_set.add(t)
        gt_per_frame.append(hi_conf)

    print(f"  GT serials: {len(gt_set)}", file=sys.stderr)

    # 5. Session 評価: 最初の duration_sec での custom model 推論
    session_frames = [(fi, ts, bgr) for (fi, ts, bgr) in all_frames if ts <= duration_sec]
    print(f"  session frames (≤ {duration_sec}s): {len(session_frames)}", file=sys.stderr)

    frame_results: list[FrameResult] = []
    custom_per_frame: list[list[tuple[str, float]]] = []
    for (fi, ts, bgr) in session_frames:
        accepted, lat_ms = _run_custom_model_on_frame(
            bgr, session, model_type, tokenizer, sw_opts,
            custom_conf_threshold, batch_size,
        )
        # GT (paddle) も同 frame の結果を流用 (= gt_per_frame の対応 entry)
        gt_idx = next((i for i, (afi, _, _) in enumerate(all_frames) if afi == fi), None)
        gt_here = gt_per_frame[gt_idx] if gt_idx is not None else []
        frame_results.append(FrameResult(
            frame_idx=fi, timestamp_sec=ts,
            custom_accepted=accepted, gt_serials=gt_here,
            inference_ms=lat_ms,
        ))
        custom_per_frame.append(accepted)

    # 6. k-of-n consensus
    custom_consensus = _aggregate_consensus(custom_per_frame, k=consensus_k)
    custom_set = set(custom_consensus.keys())

    # 7. Metrics
    intersect = custom_set & gt_set
    fps_set = custom_set - gt_set  # false positive
    miss_set = gt_set - custom_set

    recall = len(intersect) / max(1, len(gt_set))
    precision = len(intersect) / max(1, len(custom_set)) if custom_set else 0.0

    lats = [fr.inference_ms for fr in frame_results]
    mean_lat = float(np.mean(lats)) if lats else 0.0
    p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

    # 8. First complete frame: 累積 custom accepts が gt_set を覆う最初の時刻
    first_complete: float | None = None
    cumulative: dict[str, int] = {}
    for fr in frame_results:
        for text, _conf in fr.custom_accepted:
            cumulative[text] = cumulative.get(text, 0) + 1
        # consensus k に達した text を覆っているか確認
        confirmed = {t for t, n in cumulative.items() if n >= consensus_k}
        if gt_set and confirmed >= gt_set:
            first_complete = fr.timestamp_sec
            break

    # KGI Score: 誤検出があったら大きめに減点
    kgi_score = max(0.0, min(100.0, recall * 100.0 - len(fps_set) * fp_penalty))

    metrics = SessionMetrics(
        truth=gt_set,
        custom=custom_set,
        intersect=intersect,
        false_positives=fps_set,
        misses=miss_set,
        recall=recall,
        precision=precision,
        mean_frame_latency_ms=mean_lat,
        p95_frame_latency_ms=p95_lat,
        first_complete_sec=first_complete,
        n_frames=len(all_frames),
        n_session_frames=len(session_frames),
        kgi_score=kgi_score,
        kgi_fp_penalty=fp_penalty,
    )
    return metrics, frame_results


def _format_report(metrics: SessionMetrics, video_path: Path, duration_sec: float) -> str:
    lines = []
    lines.append("")
    lines.append(f"========== Session Diagnose: {video_path.name} ==========")
    lines.append(f"  GT serials (PaddleOCR 全 frame): {len(metrics.truth)}")
    lines.append(f"  Custom accepted (k-of-n consensus): {len(metrics.custom)}")
    lines.append("")

    # KGI Score: 誤検出を大きめに減点 (recall × 100 − FP × penalty)。
    # Hard Targets を満たさなくても、 改善度を 1 つのスコアで追跡できるようにする。
    fp_pen = metrics.kgi_fp_penalty
    fp_n = len(metrics.false_positives)
    fp_deduction = fp_n * fp_pen
    score_bar_len = int(metrics.kgi_score / 5)  # 100 → 20 chars
    bar = "█" * score_bar_len + "░" * (20 - score_bar_len)
    lines.append(f"  ┌─ KGI Score ────────────────────────────────────────────┐")
    lines.append(f"  │  [{bar}] {metrics.kgi_score:>6.2f} / 100              │")
    lines.append(f"  │  = recall {100*metrics.recall:>5.1f}% − FP {fp_n} × {fp_pen:.0f} (= −{fp_deduction:.0f} pt)" + " " * 6 + "│")
    lines.append(f"  └────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("  --- KGI Hard Targets ---")
    ok_recall = metrics.recall >= 0.9999  # 100%
    ok_fp = len(metrics.false_positives) == 0
    ok_miss = len(metrics.misses) == 0
    ok_session = (metrics.first_complete_sec is not None
                  and metrics.first_complete_sec <= duration_sec)
    ok_lat = metrics.mean_frame_latency_ms <= 300

    sym = lambda b: "✅" if b else "❌"  # noqa: E731
    lines.append(f"  {sym(ok_recall)} recall (= read-rate):     {100*metrics.recall:>6.2f}%  (KGI 100%)")
    lines.append(f"  {sym(ok_fp)} false-positive count:        {len(metrics.false_positives):>3}  (KGI 0)")
    lines.append(f"  {sym(ok_miss)} miss count (検出漏れ):      {len(metrics.misses):>3}  (KGI 0)")
    if metrics.first_complete_sec is not None:
        lines.append(f"  {sym(ok_session)} session 完了時間:          {metrics.first_complete_sec:>5.2f}s  "
                     f"(KGI ≤ {duration_sec:.1f}s)")
    else:
        lines.append(f"  {sym(False)} session 完了時間:          (未到達)  (KGI ≤ {duration_sec:.1f}s)")
    lines.append(f"  {sym(ok_lat)} mean frame latency:         {metrics.mean_frame_latency_ms:>5.0f}ms  "
                 f"(KGI ≤ 300ms)")
    lines.append("")
    lines.append("  --- 補助指標 ---")
    lines.append(f"  precision:                    {100*metrics.precision:>6.2f}%")
    lines.append(f"  p95 frame latency:            {metrics.p95_frame_latency_ms:>5.0f}ms")
    lines.append(f"  frame total / session frames: {metrics.n_frames} / {metrics.n_session_frames}")
    lines.append("")
    if metrics.misses:
        lines.append(f"  miss list ({len(metrics.misses)}):")
        for t in sorted(metrics.misses)[:10]:
            lines.append(f"    - {t}")
        if len(metrics.misses) > 10:
            lines.append(f"    ... +{len(metrics.misses) - 10} more")
    if metrics.false_positives:
        lines.append(f"  false-positive list ({len(metrics.false_positives)}):")
        for t in sorted(metrics.false_positives)[:10]:
            lines.append(f"    - {t}")
        if len(metrics.false_positives) > 10:
            lines.append(f"    ... +{len(metrics.false_positives) - 10} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Session-level OCR evaluation")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.3,
                        help="frame sampling fps (default 3.3 = KGI 300ms/frame)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="session duration sec (default 3.0 = KGI)")
    parser.add_argument("--custom-conf", type=float, default=0.7,
                        help="custom model conf threshold (default 0.7)")
    parser.add_argument("--paddle-conf", type=float, default=0.9,
                        help="PaddleOCR (GT) conf threshold (default 0.9)")
    parser.add_argument("--consensus-k", type=int, default=3,
                        help="k-of-n consensus: text must appear in >= k frames (default 3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--scales", type=str, default=None,
        help=(
            "sliding-window のスケール、 カンマ区切り (例: '0.7,1.0,1.4')。"
            " 未指定なら SW_DEFAULTS (= [1.0])。 runtime と整合させるなら 0.7,1.0,1.4"
        ),
    )
    parser.add_argument(
        "--fp-penalty", type=float, default=25.0,
        help="KGI Score の FP 1 件あたりの減点 (default 25.0、 4 FP で 100 点満点全消失)",
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="output metrics as JSON to this path")
    args = parser.parse_args(argv)

    sw_opts = None
    if args.scales:
        sw_opts = {"scales": [float(s.strip()) for s in args.scales.split(",")]}

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.onnx.exists():
        print(f"onnx not found: {args.onnx}", file=sys.stderr)
        return 1

    metrics, frame_results = evaluate_session(
        args.video, args.onnx,
        fps_sample=args.fps,
        duration_sec=args.duration,
        custom_conf_threshold=args.custom_conf,
        paddle_conf_threshold=args.paddle_conf,
        consensus_k=args.consensus_k,
        batch_size=args.batch_size,
        sw_opts=sw_opts,
        fp_penalty=args.fp_penalty,
    )

    report = _format_report(metrics, args.video, args.duration)
    print(report)

    if args.json:
        out = {
            "video": str(args.video),
            "onnx": str(args.onnx),
            "fps_sample": args.fps,
            "duration_sec": args.duration,
            "consensus_k": args.consensus_k,
            "truth": sorted(metrics.truth),
            "custom": sorted(metrics.custom),
            "intersect": sorted(metrics.intersect),
            "false_positives": sorted(metrics.false_positives),
            "misses": sorted(metrics.misses),
            "recall": metrics.recall,
            "precision": metrics.precision,
            "mean_frame_latency_ms": metrics.mean_frame_latency_ms,
            "p95_frame_latency_ms": metrics.p95_frame_latency_ms,
            "first_complete_sec": metrics.first_complete_sec,
            "n_frames": metrics.n_frames,
            "n_session_frames": metrics.n_session_frames,
            "kgi_score": metrics.kgi_score,
            "kgi_fp_penalty": metrics.kgi_fp_penalty,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\nJSON: {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
