"""12-head 固定長 OCR 訓練エントリ (Phase 2b)。

CRNN+CTC を 12-position fixed-length head + ∅ クラスに置き換えたモデル
(FixedHeadOCR) の訓練ループ。`train.py` の構造を踏襲しつつ:

- Loss: F.ctc_loss → F.cross_entropy (各位置で独立、reshape して 1 度の CE 呼び出し)
- Tokenizer: CTCTokenizer → FixedLengthTokenizer
- Decode: greedy_decode_with_conf → decode_with_conf (各位置 argmax)
- target shape: (sum(lengths),) → (B, FIXED_LENGTH)

Curriculum + reject 評価指標は共通モジュールから再利用。

Usage:
    python -m meiban_ocr_trainer.train_fixed_head \\
        --config configs/default.yaml \\
        [--epochs 50] [--use-rnn]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from meiban_ocr_trainer.data.augment import build_eval_transform, build_train_transform
from meiban_ocr_trainer.data.dataset import (
    RecognitionDataset,
    build_train_loader_with_ratio,
    fixed_length_collate,
    neg_ratio_for_epoch,
)
from meiban_ocr_trainer.metrics import EvaluationReport, compute_metrics, format_report
from meiban_ocr_trainer.models import FixedHeadOCR
from meiban_ocr_trainer.tokenizer import FixedLengthTokenizer
from meiban_ocr_trainer.vendors import get_vendor


def build_optimizer(model: FixedHeadOCR, cfg_train: dict) -> torch.optim.Optimizer:
    """ADDENDUM §2 と同じく LR を backbone と head で分ける AdamW。"""
    lr_backbone = float(cfg_train.get("lr_backbone", 1e-4))
    lr_head = float(cfg_train.get("lr_head", 1e-3))
    wd = float(cfg_train.get("weight_decay", 1e-4))
    head_params = list(model.classifier.parameters())
    if model.rnn is not None:
        head_params += list(model.rnn.parameters())
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )


def set_backbone_trainable(model: FixedHeadOCR, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def _build_train_loader_with_ratio_for_fixed_head(
    train_ds: RecognitionDataset,
    tokenizer: FixedLengthTokenizer,
    batch_size: int,
    neg_ratio: float,
    num_workers: int,
) -> DataLoader:
    """build_train_loader_with_ratio を fixed_length_collate に差し替えるラッパー。"""
    # build_train_loader_with_ratio は CTC collate を ctc_collate で組むので、
    # 同じ Sampler ロジックを使いつつ collate だけ差し替えるために再実装する。
    from torch.utils.data import WeightedRandomSampler

    cats = [row.get("category") or "positive" for row in train_ds.rows]
    n_pos = sum(1 for c in cats if c == "positive")
    n_neg = len(cats) - n_pos

    if n_neg == 0 or n_pos == 0:
        weights = [1.0] * len(cats)
    elif neg_ratio <= 0.0:
        weights = [1.0 if c == "positive" else 0.0 for c in cats]
    elif neg_ratio >= 1.0:
        weights = [0.0 if c == "positive" else 1.0 for c in cats]
    else:
        pos_w = (1.0 - neg_ratio) / n_pos
        neg_w = neg_ratio / n_neg
        weights = [pos_w if c == "positive" else neg_w for c in cats]

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(train_ds), replacement=True,
    )
    collate = partial(fixed_length_collate, tokenizer=tokenizer)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )


def evaluate_split(
    model: FixedHeadOCR,
    loader: DataLoader,
    tokenizer: FixedLengthTokenizer,
    device: torch.device,
    vendor_name: str = "ericsson",
    confidence_threshold: float | None = None,
) -> tuple[EvaluationReport, EvaluationReport | None, float, list[dict]]:
    """val/test を 1 周し、pattern only / pattern+conf の 2 ゲートで評価。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    preds: list[str] = []
    confs: list[float] = []
    gts: list[str] = []
    categories: list[str] = []
    subkinds: list[str] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)  # (B, fixed_length)
            logits = model(imgs)                    # (B, fixed_length, num_classes)
            # 各位置で CrossEntropy → flatten すると 1 行で書ける
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += float(loss.item())
            total_batches += 1
            for text, conf in tokenizer.decode_with_conf(logits):
                preds.append(text)
                confs.append(conf)
            gts.extend(batch["texts"])
            categories.extend(batch["categories"])
            subkinds.extend(batch["subkinds"])

    vendor = get_vendor(vendor_name)
    rep_pattern = compute_metrics(
        preds, gts, categories, subkinds, pattern=vendor.strict_regex,
    )
    rep_with_conf = None
    if confidence_threshold is not None:
        rep_with_conf = compute_metrics(
            preds, gts, categories, subkinds,
            pattern=vendor.strict_regex,
            confidences=confs, confidence_threshold=confidence_threshold,
        )
    avg_loss = total_loss / max(total_batches, 1)
    samples = [
        {"pred": p, "gt": g, "category": c, "subkind": sk, "confidence": cf}
        for p, g, c, sk, cf in zip(preds, gts, categories, subkinds, confs)
    ]
    return rep_pattern, rep_with_conf, avg_loss, samples


def train_loop(cfg: dict, output_dir: Path, use_rnn: bool = False) -> dict:
    device = torch.device(cfg["runtime"].get("device", "cpu"))
    torch.manual_seed(int(cfg["runtime"].get("seed", 42)))
    tokenizer = FixedLengthTokenizer()

    data_root = Path(cfg["data"]["root"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 4))
    confidence_threshold = cfg["train"].get("confidence_threshold")
    if confidence_threshold is not None:
        confidence_threshold = float(confidence_threshold)
    neg_ratio_schedule = cfg["train"].get("neg_ratio_schedule") or []
    val_vendor = cfg.get("data", {}).get("vendor", "ericsson")

    train_ds = RecognitionDataset(
        data_root, "train", transform=build_train_transform(),
    )
    val_ds = RecognitionDataset(
        data_root, "val", transform=build_eval_transform(),
    )
    test_ds = RecognitionDataset(
        data_root, "test", transform=build_eval_transform(),
    )

    eval_collate = partial(fixed_length_collate, tokenizer=tokenizer)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(0, num_workers // 2), collate_fn=eval_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=eval_collate,
    )

    model = FixedHeadOCR(
        use_rnn=use_rnn,
        rnn_hidden=int(cfg["model"].get("rnn_hidden", 64)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pretrained=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_fixed_head] model params: {n_params:,} (use_rnn={use_rnn})",
          file=sys.stderr)

    optimizer = build_optimizer(model, cfg["train"])
    epochs = int(cfg["train"]["epochs"])
    freeze_warmup = int(cfg["train"].get("freeze_backbone_epochs", 2))
    patience = int(cfg["train"].get("early_stopping_patience", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_cer = math.inf
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    epochs_since_improvement = 0

    n_train_total = len(train_ds)
    n_train_pos = sum(
        1 for r in train_ds.rows if (r.get("category") or "positive") == "positive"
    )
    n_train_neg = n_train_total - n_train_pos
    print(
        f"[train_fixed_head] start. epochs={epochs}, train: {n_train_total} "
        f"(pos {n_train_pos} / neg {n_train_neg}), val={len(val_ds)}, "
        f"test={len(test_ds)}",
        file=sys.stderr,
    )
    if neg_ratio_schedule:
        print(f"[train_fixed_head] curriculum: {neg_ratio_schedule}", file=sys.stderr)

    for epoch in range(1, epochs + 1):
        set_backbone_trainable(model, epoch > freeze_warmup)

        neg_ratio = (
            neg_ratio_for_epoch(neg_ratio_schedule, epoch)
            if neg_ratio_schedule else 0.0
        )
        train_loader = _build_train_loader_with_ratio_for_fixed_head(
            train_ds, tokenizer,
            batch_size=batch_size, neg_ratio=neg_ratio, num_workers=num_workers,
        )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)  # (B, fixed_length)
            logits = model(imgs)                    # (B, fixed_length, num_classes)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        val_rep_p, val_rep_c, val_loss, _ = evaluate_split(
            model, val_loader, tokenizer, device,
            vendor_name=val_vendor,
            confidence_threshold=confidence_threshold,
        )
        val_cer_effective = val_rep_p.cer if val_rep_p.cer is not None else 1.0
        val_em_effective = val_rep_p.em if val_rep_p.em is not None else 0.0
        dt = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]
        improved = val_cer_effective < best_val_cer - 1e-6

        print(
            f"[epoch {epoch:3d}/{epochs}] train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_CER={val_cer_effective:.4f}  "
            f"val_EM={val_em_effective:.3f}  neg_ratio={neg_ratio:.2f}  "
            f"lr={lr_now:.2e}  dt={dt:.1f}s"
            + ("  *best*" if improved else ""),
            file=sys.stderr,
        )
        if val_rep_p.n_neg > 0:
            print(format_report(val_rep_p, label=f"epoch {epoch} val [gate=pattern]"),
                  file=sys.stderr)
            if val_rep_c is not None:
                print(
                    format_report(
                        val_rep_c,
                        label=f"epoch {epoch} val [gate=pattern+conf>={confidence_threshold}]",
                    ),
                    file=sys.stderr,
                )

        history.append({
            "epoch": epoch,
            "neg_ratio": neg_ratio,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer_effective,
            "val_em": val_em_effective,
            "val_metrics_pattern": val_rep_p.to_dict(),
            "val_metrics_pattern_and_conf": (
                val_rep_c.to_dict() if val_rep_c is not None else None
            ),
            "lr_head": lr_now,
            "dt_sec": dt,
        })

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_cer": val_cer_effective,
            "config": cfg,
            "model_type": "fixed_head",
            "use_rnn": use_rnn,
        }, last_path)
        if improved:
            best_val_cer = val_cer_effective
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_cer": val_cer_effective,
                "config": cfg,
                "model_type": "fixed_head",
                "use_rnn": use_rnn,
            }, best_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(
                    f"[train_fixed_head] early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs)",
                    file=sys.stderr,
                )
                break

    print(
        f"\n[train_fixed_head] loading best ckpt (epoch {best_epoch}, "
        f"val_CER {best_val_cer:.4f})...", file=sys.stderr,
    )
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_rep_p, test_rep_c, test_loss, test_samples = evaluate_split(
        model, test_loader, tokenizer, device,
        vendor_name=val_vendor,
        confidence_threshold=confidence_threshold,
    )
    test_cer = test_rep_p.cer if test_rep_p.cer is not None else 1.0
    test_em = test_rep_p.em if test_rep_p.em is not None else 0.0
    print(format_report(test_rep_p, label="test [gate=pattern]"), file=sys.stderr)
    if test_rep_c is not None:
        print(
            format_report(
                test_rep_c,
                label=f"test [gate=pattern+conf>={confidence_threshold}]",
            ),
            file=sys.stderr,
        )
    print(f"[train_fixed_head] test_loss={test_loss:.4f}", file=sys.stderr)
    print("[train_fixed_head] test predictions:", file=sys.stderr)
    for s in test_samples:
        tag = "POS" if s["category"] == "positive" else "NEG"
        print(
            f"    [{tag}] pred={s['pred']!r:20s} gt={s['gt']!r:20s} "
            f"conf={s['confidence']:.3f}",
            file=sys.stderr,
        )

    summary = {
        "model_type": "fixed_head",
        "use_rnn": use_rnn,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_cer": best_val_cer,
        "test_cer": test_cer,
        "test_em": test_em,
        "test_loss": test_loss,
        "test_metrics_pattern": test_rep_p.to_dict(),
        "test_metrics_pattern_and_conf": (
            test_rep_c.to_dict() if test_rep_c is not None else None
        ),
        "confidence_threshold": confidence_threshold,
        "neg_ratio_schedule": neg_ratio_schedule,
        "test_samples": test_samples,
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="meiban-ocr 12-head training")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: runs/YYYYMMDD-HHMMSS_fh")
    parser.add_argument("--use-rnn", action="store_true",
                        help="enable optional BiGRU (default: off, smaller params)")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"[train_fixed_head] config not found: {args.config}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    output_dir = args.output_dir or (
        Path(cfg["output"]["runs_dir"]) / (time.strftime("%Y%m%d-%H%M%S") + "_fh")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
    )

    summary = train_loop(cfg, output_dir, use_rnn=args.use_rnn)
    print(f"\n[train_fixed_head] run dir: {output_dir}", file=sys.stderr)
    print(
        f"[train_fixed_head] best_val_CER={summary['best_val_cer']:.4f}  "
        f"test_CER={summary['test_cer']:.4f}  test_EM={summary['test_em']:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
