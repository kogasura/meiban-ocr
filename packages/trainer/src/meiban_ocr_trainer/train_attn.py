"""attention decoder (CRNNAttn) の訓練エントリ。

train.py (CTC) との違い:
- model = CRNNAttn。訓練済み CTC checkpoint (model.init_from) から encoder を warm start
- loss = CTC (補助) + λ·CrossEntropy (attention、teacher forcing)
- val/test の主指標は attention decode の EM/CER。CTC head の EM も退行検知用に併記
- 最初の freeze_encoder_epochs は encoder 凍結 (decoder warmup)

Usage:
    python -m meiban_ocr_trainer.train_attn --config configs/crnn_attn.yaml \\
        --output-dir runs/cr_attn
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

from meiban_ocr_trainer.constants import BLANK_IDX
from meiban_ocr_trainer.data.augment import build_eval_transform, build_train_transform
from meiban_ocr_trainer.data.dataset import (
    RecognitionDataset,
    build_train_loader_with_ratio,
    ctc_collate,
    neg_ratio_for_epoch,
)
from meiban_ocr_trainer.metrics import compute_metrics, format_report
from meiban_ocr_trainer.models.crnn_attn import CRNNAttn
from meiban_ocr_trainer.tokenizer import AttnTokenizer, CTCTokenizer
from meiban_ocr_trainer.vendors import get_vendor


def joint_loss(
    ctc_logits: torch.Tensor,
    attn_logits: torch.Tensor,
    batch: dict,
    attn_targets: torch.Tensor,
    lambda_attn: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """CTC (補助) + λ·CE (attention)。返り値 (total, ctc_item, attn_item)。"""
    log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    input_lengths = torch.full(
        (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long, device=device
    )
    ctc = F.ctc_loss(
        log_probs, batch["targets"].to(device), input_lengths,
        batch["target_lengths"].to(device), blank=BLANK_IDX, zero_infinity=True,
    )
    ce = F.cross_entropy(
        attn_logits.reshape(-1, attn_logits.size(-1)),
        attn_targets.reshape(-1),
        ignore_index=AttnTokenizer.IGNORE_INDEX,
    )
    total = ctc + lambda_attn * ce
    return total, float(ctc.item()), float(ce.item())


@torch.no_grad()
def evaluate_split(
    model: CRNNAttn,
    loader: DataLoader,
    attn_tok: AttnTokenizer,
    ctc_tok: CTCTokenizer,
    device: torch.device,
    vendor_name: str,
    confidence_threshold: float | None,
):
    """attention decode を主指標に評価。CTC head の EM も併記する。"""
    model.eval()
    preds, confs, gts, cats, subs = [], [], [], [], []
    ctc_preds = []
    for batch in loader:
        imgs = batch["images"].to(device)
        ctc_logits, attn_logits = model(imgs, teacher_inputs=None)
        for text, conf in attn_tok.decode_with_conf(attn_logits):
            preds.append(text)
            confs.append(conf)
        ctc_preds.extend(ctc_tok.greedy_decode(ctc_logits))
        gts.extend(batch["texts"])
        cats.extend(batch["categories"])
        subs.extend(batch["subkinds"])

    vendor = get_vendor(vendor_name)
    rep = compute_metrics(preds, gts, cats, subs, pattern=vendor.strict_regex)
    rep_conf = None
    if confidence_threshold is not None:
        rep_conf = compute_metrics(
            preds, gts, cats, subs, pattern=vendor.strict_regex,
            confidences=confs, confidence_threshold=confidence_threshold,
        )
    rep_ctc = compute_metrics(ctc_preds, gts, cats, subs, pattern=vendor.strict_regex)
    samples = [
        {"pred": p, "gt": g, "category": c, "subkind": sk, "confidence": cf}
        for p, g, c, sk, cf in zip(preds, gts, cats, subs, confs)
    ]
    return rep, rep_conf, rep_ctc, samples


def train_loop(cfg: dict, output_dir: Path) -> dict:
    device = torch.device(cfg["runtime"].get("device", "cpu"))
    torch.manual_seed(int(cfg["runtime"].get("seed", 42)))
    ctc_tok = CTCTokenizer()
    attn_tok = AttnTokenizer()

    data_root = Path(cfg["data"]["root"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 4))
    confidence_threshold = cfg["train"].get("confidence_threshold")
    if confidence_threshold is not None:
        confidence_threshold = float(confidence_threshold)
    neg_ratio_schedule = cfg["train"].get("neg_ratio_schedule") or []
    vendor_name = cfg.get("data", {}).get("vendor", "ericsson")
    labels_filename = cfg.get("data", {}).get("labels_filename", "labels.tsv")
    resize_mode = cfg.get("data", {}).get("resize_mode", "stretch")
    lambda_attn = float(cfg["train"].get("lambda_attn", 1.0))

    train_ds = RecognitionDataset(
        data_root, "train", ctc_tok, build_train_transform(resize_mode),
        labels_filename=labels_filename,
    )
    val_ds = RecognitionDataset(
        data_root, "val", ctc_tok, build_eval_transform(resize_mode),
        labels_filename=labels_filename,
    )
    test_ds = RecognitionDataset(
        data_root, "test", ctc_tok, build_eval_transform(resize_mode),
        labels_filename=labels_filename,
    )
    eval_collate = partial(ctc_collate, tokenizer=ctc_tok)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(0, num_workers // 2), collate_fn=eval_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=eval_collate,
    )

    init_from = cfg["model"].get("init_from")
    if init_from and Path(init_from).exists():
        print(f"[train_attn] warm start from {init_from}", file=sys.stderr)
        model = CRNNAttn.from_ctc_checkpoint(
            init_from, hidden_size=int(cfg["model"].get("crnn_hidden_size", 256)),
        ).to(device)
    else:
        raise FileNotFoundError(
            f"crnn_attn requires model.init_from (訓練済み CTC ckpt), got: {init_from!r}"
        )

    wd = float(cfg["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.get_param_groups(
        backbone_lr=float(cfg["train"].get("backbone_lr", 1e-5)),
        rnn_lr=float(cfg["train"].get("rnn_lr", 5e-5)),
        ctc_head_lr=float(cfg["train"].get("ctc_head_lr", 1e-4)),
        decoder_lr=float(cfg["train"].get("decoder_lr", 1e-3)),
        weight_decay=wd,
    ))
    epochs = int(cfg["train"]["epochs"])
    freeze_warmup = int(cfg["train"].get("freeze_encoder_epochs", 2))
    patience = int(cfg["train"].get("early_stopping_patience", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_cer = math.inf
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    since_improve = 0

    n_pos = sum(1 for r in train_ds.rows if (r.get("category") or "positive") == "positive")
    print(
        f"[train_attn] start. epochs={epochs}, train={len(train_ds)} "
        f"(pos {n_pos} / neg {len(train_ds) - n_pos}), val={len(val_ds)}, "
        f"test={len(test_ds)}, lambda_attn={lambda_attn}",
        file=sys.stderr,
    )

    for epoch in range(1, epochs + 1):
        model.freeze_encoder(freeze=epoch <= freeze_warmup)
        neg_ratio = neg_ratio_for_epoch(neg_ratio_schedule, epoch) if neg_ratio_schedule else 0.0
        train_loader = build_train_loader_with_ratio(
            train_ds, ctc_tok, batch_size=batch_size,
            neg_ratio=neg_ratio, num_workers=num_workers,
        )

        model.train()
        ep_total = ep_ctc = ep_attn = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            imgs = batch["images"].to(device)
            teacher_inputs, attn_targets = attn_tok.encode_teacher(batch["texts"])
            teacher_inputs = teacher_inputs.to(device)
            attn_targets = attn_targets.to(device)
            ctc_logits, attn_logits = model(imgs, teacher_inputs=teacher_inputs)
            loss, ctc_item, attn_item = joint_loss(
                ctc_logits, attn_logits, batch, attn_targets, lambda_attn, device,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_total += float(loss.item())
            ep_ctc += ctc_item
            ep_attn += attn_item
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)

        rep, rep_conf, rep_ctc, _ = evaluate_split(
            model, val_loader, attn_tok, ctc_tok, device,
            vendor_name, confidence_threshold,
        )
        val_cer = rep.cer if rep.cer is not None else 1.0
        val_em = rep.em if rep.em is not None else 0.0
        ctc_em = rep_ctc.em if rep_ctc.em is not None else 0.0
        # best 選定は CER + 偽発火ペナルティの合成。CER のみだと「負例未学習だが
        # positive は読める」epoch が best になり、reject 不能モデルを保存してしまう
        # (v1 で実際に発生: best=epoch5 が neg_fire 99%)。
        val_fpr = rep.fpr_pattern if rep.fpr_pattern is not None else 0.0
        val_score = val_cer + 0.1 * val_fpr
        improved = val_score < best_val_cer - 1e-6
        dt = time.time() - t0
        print(
            f"[epoch {epoch:3d}/{epochs}] loss={ep_total/nb:.4f} "
            f"(ctc={ep_ctc/nb:.4f} attn={ep_attn/nb:.4f})  "
            f"val_CER={val_cer:.4f}  val_EM={val_em:.3f}  (ctc_EM={ctc_em:.3f})  "
            f"val_FPR={val_fpr:.4f}  score={val_score:.4f}  "
            f"neg_ratio={neg_ratio:.2f}  dt={dt:.1f}s" + ("  *best*" if improved else ""),
            file=sys.stderr,
        )
        if rep.n_neg > 0:
            print(format_report(rep, label=f"epoch {epoch} val [attn, gate=pattern]"),
                  file=sys.stderr)

        history.append({
            "epoch": epoch, "neg_ratio": neg_ratio,
            "train_loss": ep_total / nb, "train_ctc": ep_ctc / nb,
            "train_attn": ep_attn / nb,
            "val_cer": val_cer, "val_em": val_em, "val_ctc_em": ctc_em,
            "val_metrics": rep.to_dict(), "dt_sec": dt,
        })

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_cer": val_cer, "config": cfg,
        }, last_path)
        if improved:
            best_val_cer = val_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_cer": val_cer, "config": cfg,
            }, best_path)
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                print(f"[train_attn] early stopping at epoch {epoch}", file=sys.stderr)
                break

    print(f"\n[train_attn] loading best (epoch {best_epoch}, val_CER {best_val_cer:.4f})",
          file=sys.stderr)
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    rep, rep_conf, rep_ctc, samples = evaluate_split(
        model, test_loader, attn_tok, ctc_tok, device,
        vendor_name, confidence_threshold,
    )
    test_cer = rep.cer if rep.cer is not None else 1.0
    test_em = rep.em if rep.em is not None else 0.0
    print(format_report(rep, label="test [attn, gate=pattern]"), file=sys.stderr)
    print(format_report(rep_ctc, label="test [aux CTC head]"), file=sys.stderr)

    summary = {
        "arch": "crnn_attn",
        "best_epoch": best_epoch,
        "best_val_cer": best_val_cer,
        "test_cer": test_cer,
        "test_em": test_em,
        "test_metrics_pattern": rep.to_dict(),
        "test_metrics_ctc_head": rep_ctc.to_dict(),
        "test_metrics_pattern_and_conf": rep_conf.to_dict() if rep_conf else None,
        "confidence_threshold": confidence_threshold,
        "neg_ratio_schedule": neg_ratio_schedule,
        "test_samples": samples,
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="meiban-ocr attention decoder training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    output_dir = args.output_dir or Path(cfg["output"]["runs_dir"]) / time.strftime(
        "%Y%m%d-%H%M%S_attn"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    summary = train_loop(cfg, output_dir)
    print(f"\n[train_attn] run dir: {output_dir}", file=sys.stderr)
    print(
        f"[train_attn] best_val_CER={summary['best_val_cer']:.4f}  "
        f"test_CER={summary['test_cer']:.4f}  test_EM={summary['test_em']:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
