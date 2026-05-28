"""評価エントリ。指定 checkpoint + split に対し CER / WER / EM を計算。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from meiban_ocr_trainer.constants import BLANK_IDX
from meiban_ocr_trainer.data.augment import build_eval_transform
from meiban_ocr_trainer.data.dataset import RecognitionDataset, ctc_collate
from meiban_ocr_trainer.models import TinyOCRModel
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def _compute_metrics(preds: list[str], targets: list[str]) -> dict[str, float]:
    from torchmetrics.text import CharErrorRate, WordErrorRate

    cer = float(CharErrorRate()(preds, targets))
    wer = float(WordErrorRate()(preds, targets))
    em = sum(p == t for p, t in zip(preds, targets)) / max(len(targets), 1)
    return {"cer": cer, "wer": wer, "em": em}


def main(argv: list[str] | None = None) -> int:
    from functools import partial

    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="meiban-ocr evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/recognition"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    tokenizer = CTCTokenizer()
    ds = RecognitionDataset(args.data_root, args.split, tokenizer, build_eval_transform())
    collate = partial(ctc_collate, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    model.eval()

    preds: list[str] = []
    gts: list[str] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(imgs)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full(
                (imgs.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=BLANK_IDX, zero_infinity=True,
            )
            total_loss += float(loss.item())
            n_batches += 1
            preds.extend(tokenizer.greedy_decode(logits))
            gts.extend(batch["texts"])

    metrics = _compute_metrics(preds, gts)
    metrics["avg_loss"] = total_loss / max(n_batches, 1)
    metrics["n_samples"] = len(gts)
    print(json.dumps(metrics, indent=2), file=sys.stderr)
    if args.save_predictions:
        args.save_predictions.write_text(
            json.dumps(
                {"metrics": metrics,
                 "predictions": [{"pred": p, "gt": g, "correct": p == g}
                                 for p, g in zip(preds, gts)]},
                indent=2, ensure_ascii=False,
            ), encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
