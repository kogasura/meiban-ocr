"""評価エントリ。指定 checkpoint + split に対し新指標 (CER/EM/FPR/RR) を計算。

訓練後に閾値スイープ (e.g., conf 0.5 / 0.6 / 0.7 / 0.8 / 0.9) を行って、
FPR_pattern vs AcceptanceRecall のトレードオフを可視化するのにも使う。

Usage:
    python -m meiban_ocr_trainer.evaluate \\
        --checkpoint runs/<run>/best.pt --split val \\
        [--confidence-threshold 0.7] \\
        [--threshold-sweep 0.5,0.6,0.7,0.8,0.9] \\
        [--save-predictions out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from meiban_ocr_trainer.data.augment import build_eval_transform
from meiban_ocr_trainer.data.dataset import RecognitionDataset, ctc_collate
from meiban_ocr_trainer.metrics import format_report
from meiban_ocr_trainer.models import TinyOCRModel
from meiban_ocr_trainer.tokenizer import CTCTokenizer
from meiban_ocr_trainer.train import evaluate_split


MAX_SWEEP_THRESHOLDS = 100


def _parse_threshold_list(s: str | None) -> list[float]:
    """カンマ区切り floats をパース。0-1 範囲外 / 100 個超 / 不正な値で ValueError。"""
    if not s:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) > MAX_SWEEP_THRESHOLDS:
        raise ValueError(
            f"too many thresholds ({len(parts)} > {MAX_SWEEP_THRESHOLDS})"
        )
    out: list[float] = []
    for p in parts:
        v = float(p)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold out of [0, 1]: {v}")
        out.append(v)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="meiban-ocr evaluation (v2 metrics)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/recognition"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--vendor", type=str, default="ericsson",
        help="strict_regex に使う vendor 名 (vendors.py 参照)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="pattern + conf>=threshold ゲートで評価。指定無しなら pattern only。",
    )
    parser.add_argument(
        "--threshold-sweep", type=str, default=None,
        help="複数閾値のスイープ。例: '0.5,0.6,0.7,0.8,0.9'",
    )
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    tokenizer = CTCTokenizer()
    ds = RecognitionDataset(args.data_root, args.split, tokenizer, build_eval_transform())
    collate = partial(ctc_collate, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Why weights_only=True: pickle 経由の任意コード実行を防ぐ。
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = TinyOCRModel(
        num_classes=int(model_cfg.get("num_classes", 37)),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    thresholds = _parse_threshold_list(args.threshold_sweep)
    if args.confidence_threshold is not None and args.confidence_threshold not in thresholds:
        thresholds.append(args.confidence_threshold)
    thresholds = sorted(set(thresholds))

    # まず pattern only (always)
    rep_p, _rep_c_unused, avg_loss, samples = evaluate_split(
        model, dl, tokenizer, device,
        vendor_name=args.vendor,
        confidence_threshold=None,
    )
    print(format_report(rep_p, label=f"{args.split} [gate=pattern]"), file=sys.stderr)
    print(f"  avg_loss={avg_loss:.4f}", file=sys.stderr)

    sweep_results: list[dict] = []
    for th in thresholds:
        rep_c = _recompute_with_threshold(samples, args.vendor, th)
        print(format_report(rep_c, label=f"{args.split} [gate=pattern+conf>={th}]"),
              file=sys.stderr)
        sweep_results.append({"threshold": th, "metrics": rep_c.to_dict()})

    if args.save_predictions:
        out = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "vendor": args.vendor,
            "avg_loss": avg_loss,
            "metrics_pattern": rep_p.to_dict(),
            "metrics_sweep": sweep_results,
            "predictions": samples,
        }
        args.save_predictions.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    return 0


def _recompute_with_threshold(
    samples: list[dict], vendor: str, threshold: float,
) -> "EvaluationReport":  # noqa: F821
    """既に取得済みの (pred, conf) リストから別 threshold で再計算 (forward 不要)。"""
    from meiban_ocr_trainer.metrics import compute_metrics
    from meiban_ocr_trainer.vendors import get_vendor

    preds = [s["pred"] for s in samples]
    gts = [s["gt"] for s in samples]
    cats = [s["category"] for s in samples]
    subkinds = [s["subkind"] for s in samples]
    confs = [s["confidence"] for s in samples]
    return compute_metrics(
        preds, gts, cats, subkinds,
        pattern=get_vendor(vendor).strict_regex,
        confidences=confs, confidence_threshold=threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
