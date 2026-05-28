"""訓練エントリ。HANDOFF.md Phase 1 + ADDENDUM.md §2 を実装。

主要設計:
- バックボーン (MobileNetV3-Small) は ImageNet pretrained を継承 (ADDENDUM §1)
- 学習率差分: backbone=1e-4 / RNN=1e-3 / classifier=1e-3 (ADDENDUM §2)
- 最初の `freeze_backbone_epochs` だけ backbone を freeze し、頭部のみ訓練するウォームアップ
- CTC loss、Adam-W、Cosine LR schedule
- val CER で best ckpt を保存、early stopping (patience エポック)

Usage:
    python -m meiban_ocr_trainer.train --config configs/default.yaml [--epochs 50]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from meiban_ocr_trainer.constants import BLANK_IDX
from meiban_ocr_trainer.data.dataset import build_dataloaders
from meiban_ocr_trainer.models import TinyOCRModel
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def _compute_cer(preds: list[str], targets: list[str]) -> float:
    """文字単位 編集距離 / 参照長 の平均。Lazy import で torchmetrics 依存。"""
    from torchmetrics.text import CharErrorRate

    cer = CharErrorRate()
    return float(cer(preds, targets))


def _compute_em(preds: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets)) / len(targets)


def build_optimizer(model: TinyOCRModel, cfg_train: dict) -> torch.optim.Optimizer:
    """ADDENDUM §2 に従い LR を 3 グループに分けた AdamW。"""
    lr_backbone = float(cfg_train.get("lr_backbone", 1e-4))
    lr_head = float(cfg_train.get("lr_head", 1e-3))
    wd = float(cfg_train.get("weight_decay", 1e-4))
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.rnn.parameters(), "lr": lr_head},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )


def set_backbone_trainable(model: TinyOCRModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def evaluate_split(
    model: TinyOCRModel,
    loader: DataLoader,
    tokenizer: CTCTokenizer,
    device: torch.device,
) -> tuple[float, float, float, list[tuple[str, str]]]:
    """val/test を1周し (CER, EM, avg_loss, [(pred, gt), ...]) を返す。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    preds: list[str] = []
    gts: list[str] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(imgs)  # (B, T, C)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full(
                (imgs.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=BLANK_IDX, zero_infinity=True,
            )
            total_loss += float(loss.item())
            total_batches += 1
            batch_preds = tokenizer.greedy_decode(logits)
            preds.extend(batch_preds)
            gts.extend(batch["texts"])
    cer = _compute_cer(preds, gts) if preds else 1.0
    em = _compute_em(preds, gts)
    avg_loss = total_loss / max(total_batches, 1)
    return cer, em, avg_loss, list(zip(preds, gts))


def train_loop(cfg: dict, output_dir: Path) -> dict:
    device = torch.device(cfg["runtime"].get("device", "cpu"))
    torch.manual_seed(int(cfg["runtime"].get("seed", 42)))
    tokenizer = CTCTokenizer()

    # Data
    data_root = Path(cfg["data"]["root"])
    loaders = build_dataloaders(
        data_root,
        tokenizer,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )

    # Model
    model = TinyOCRModel(
        num_classes=int(cfg["model"]["num_classes"]),
        rnn_hidden=int(cfg["model"]["rnn_hidden"]),
        rnn_layers=int(cfg["model"]["rnn_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        pretrained=True,
    ).to(device)

    optimizer = build_optimizer(model, cfg["train"])
    epochs = int(cfg["train"]["epochs"])
    freeze_warmup = int(cfg["train"].get("freeze_backbone_epochs", 2))
    patience = int(cfg["train"].get("early_stopping_patience", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    history = []
    best_val_cer = math.inf
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    epochs_since_improvement = 0
    n_train = len(loaders["train"].dataset)
    print(f"[train] start. epochs={epochs}, train_size={n_train}, "
          f"val_size={len(loaders['val'].dataset)}, test_size={len(loaders['test'].dataset)}",
          file=sys.stderr)

    for epoch in range(1, epochs + 1):
        # Warmup: 最初の freeze_warmup エポックは backbone を凍結
        set_backbone_trainable(model, epoch > freeze_warmup)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in loaders["train"]:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(imgs)  # (B, T, C)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full(
                (imgs.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=BLANK_IDX, zero_infinity=True,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_cer, val_em, val_loss, val_samples = evaluate_split(
            model, loaders["val"], tokenizer, device
        )
        dt = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]  # head LR
        improved = val_cer < best_val_cer - 1e-6

        print(
            f"[epoch {epoch:3d}/{epochs}] train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_CER={val_cer:.4f}  val_EM={val_em:.3f}  "
            f"lr={lr_now:.2e}  dt={dt:.1f}s"
            + ("  *best*" if improved else ""),
            file=sys.stderr,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "val_em": val_em,
            "lr_head": lr_now,
            "dt_sec": dt,
        })

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_cer": val_cer,
            "config": cfg,
        }, last_path)
        if improved:
            best_val_cer = val_cer
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_cer": val_cer,
                "config": cfg,
            }, best_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs)", file=sys.stderr)
                break

    # Final test evaluation with best ckpt
    print(f"\n[train] loading best ckpt (epoch {best_epoch}, val_CER {best_val_cer:.4f})...",
          file=sys.stderr)
    # Why weights_only=True: pickle 経由の任意コード実行を防ぐ。
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_cer, test_em, test_loss, test_samples = evaluate_split(
        model, loaders["test"], tokenizer, device
    )
    print(f"[train] test_CER={test_cer:.4f}  test_EM={test_em:.3f}  test_loss={test_loss:.4f}",
          file=sys.stderr)
    print("[train] test predictions:", file=sys.stderr)
    for pred, gt in test_samples:
        print(f"    pred={pred!r:20s} gt={gt!r}", file=sys.stderr)

    summary = {
        "best_epoch": best_epoch,
        "best_val_cer": best_val_cer,
        "test_cer": test_cer,
        "test_em": test_em,
        "test_loss": test_loss,
        "test_samples": [{"pred": p, "gt": g} for p, g in test_samples],
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="meiban-ocr training")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, help="override epochs", default=None)
    parser.add_argument("--batch-size", type=int, help="override batch size", default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: runs/YYYYMMDD-HHMMSS")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"[train] config not found: {args.config}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(args.config.read_text())

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    output_dir = args.output_dir or Path(cfg["output"]["runs_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    summary = train_loop(cfg, output_dir)
    print(f"\n[train] run dir: {output_dir}", file=sys.stderr)
    print(f"[train] best_val_CER={summary['best_val_cer']:.4f}  "
          f"test_CER={summary['test_cer']:.4f}  test_EM={summary['test_em']:.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
