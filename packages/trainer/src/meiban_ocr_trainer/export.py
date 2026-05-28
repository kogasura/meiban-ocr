"""PyTorch checkpoint → ONNX エクスポート + onnx-simplifier + 動的 INT8 量子化。

HANDOFF.md Phase 2:
- opset_version=17 (onnxruntime-web 1.19+ 互換)
- dynamic batch dimension
- onnx-simplifier でグラフ簡略化
- onnxruntime.quantization.quantize_dynamic で重み INT8 化
- 量子化前後の CER 差を検証 (< 0.2% を要件とする、HANDOFF.md Phase 2 DoD)

Usage:
    python -m meiban_ocr_trainer.export \\
        --checkpoint runs/20260527-191732/best.pt \\
        --output-dir models/ \\
        --name meiban-ocr-v1 \\
        --validate-data data/recognition
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from meiban_ocr_trainer.constants import INPUT_HEIGHT, INPUT_WIDTH
from meiban_ocr_trainer.models import TinyOCRModel
from meiban_ocr_trainer.tokenizer import CTCTokenizer


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> TinyOCRModel:
    cfg = ckpt.get("config", {}).get("model", {})
    model = TinyOCRModel(
        num_classes=int(cfg.get("num_classes", 37)),
        rnn_hidden=int(cfg.get("rnn_hidden", 128)),
        rnn_layers=int(cfg.get("rnn_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        pretrained=False,  # weights は ckpt から復元するので不要
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    return model


def export_onnx(
    model: TinyOCRModel,
    output_path: Path,
    opset: int = 17,
    height: int = INPUT_HEIGHT,
    width: int = INPUT_WIDTH,
) -> None:
    """PyTorch → ONNX エクスポート。"""
    model.eval()
    dummy = torch.zeros(1, 1, height, width, dtype=torch.float32)
    print(f"[export] exporting to ONNX (opset={opset})...", file=sys.stderr)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(
        f"[export] wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)",
        file=sys.stderr,
    )


def simplify_onnx(input_path: Path, output_path: Path) -> None:
    """onnx-simplifier でグラフを簡略化。"""
    import onnx
    from onnxsim import simplify

    print("[export] running onnx-simplifier...", file=sys.stderr)
    model = onnx.load(str(input_path))
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnx-simplifier could not validate the simplified model")
    onnx.save(simplified, str(output_path))
    print(
        f"[export] simplified → {output_path} ({output_path.stat().st_size / 1024:.1f} KB)",
        file=sys.stderr,
    )


def quantize_fp16(input_path: Path, output_path: Path) -> None:
    """FP16 化。重みと中間活性化を半精度に変換 → サイズ約 50% 減。

    Why FP16 over INT8: `quantize_dynamic` の INT8 出力は `ConvInteger` 演算子を
    生成するが、onnxruntime CPU EP も onnxruntime-web も完全サポートしておらず
    実行時エラーが出る (`NOT_IMPLEMENTED: Could not find an implementation for
    ConvInteger`)。FP16 化は ONNX 標準ノードを使うため WebGPU/WASM 双方で動く。
    精度劣化は通常無視できる範囲 (CRNN 程度なら CER 差は 0.001 未満)。
    """
    import onnx
    from onnxconverter_common import float16

    print("[export] FP16 quantization...", file=sys.stderr)
    model_fp32 = onnx.load(str(input_path))
    # Why keep_io_types=True: 入力/出力はFP32のまま (前処理/後処理を変更しない)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))
    print(
        f"[export] FP16 → {output_path} ({output_path.stat().st_size / 1024:.1f} KB)",
        file=sys.stderr,
    )


def evaluate_onnx_cer(
    onnx_path: Path,
    data_root: Path,
    split: str = "val",
    batch_size: int = 32,
) -> dict[str, float]:
    """ONNX モデルを Python onnxruntime で読み込み、指定 split で CER/EM を計算。"""
    from functools import partial

    import numpy as np
    import onnxruntime as ort
    from torch.utils.data import DataLoader
    from torchmetrics.text import CharErrorRate

    from meiban_ocr_trainer.data.augment import build_eval_transform
    from meiban_ocr_trainer.data.dataset import RecognitionDataset, ctc_collate

    tokenizer = CTCTokenizer()
    ds = RecognitionDataset(data_root, split, tokenizer, build_eval_transform())
    collate = partial(ctc_collate, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    preds_all: list[str] = []
    gts_all: list[str] = []
    for batch in dl:
        x = batch["images"].numpy().astype(np.float32)
        (logits,) = sess.run(None, {input_name: x})  # (B, T, C)
        logits_t = torch.from_numpy(logits)
        preds = tokenizer.greedy_decode(logits_t)
        preds_all.extend(preds)
        gts_all.extend(batch["texts"])

    cer = float(CharErrorRate()(preds_all, gts_all))
    em = sum(p == g for p, g in zip(preds_all, gts_all)) / max(len(gts_all), 1)
    return {"cer": cer, "em": em, "n_samples": len(gts_all)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX (+ INT8)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--name", type=str, default="meiban-ocr-v1",
                        help="output basename (.onnx is appended)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-quantize", action="store_true",
                        help="skip FP16 quantization")
    parser.add_argument("--validate-data", type=Path, default=None,
                        help="data/recognition root; if given, run CER comparison")
    parser.add_argument("--validate-split", type=str, default="val")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / f"{args.name}.fp32.onnx"
    simplified_path = args.output_dir / f"{args.name}.fp32.sim.onnx"
    fp16_path = args.output_dir / f"{args.name}.fp16.onnx"
    final_path = args.output_dir / f"{args.name}.onnx"

    device = torch.device("cpu")
    print(f"[export] loading checkpoint: {args.checkpoint}", file=sys.stderr)
    # Why weights_only=True: pickle 経由の任意コード実行を防ぐ。
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = _build_model_from_ckpt(ckpt, device)
    print(f"  ckpt epoch={ckpt.get('epoch')}, val_CER={ckpt.get('val_cer')}",
          file=sys.stderr)

    export_onnx(model, fp32_path, opset=args.opset)
    simplify_onnx(fp32_path, simplified_path)
    if args.no_quantize:
        shutil.copyfile(simplified_path, final_path)
    else:
        quantize_fp16(simplified_path, fp16_path)
        shutil.copyfile(fp16_path, final_path)

    report: dict = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "opset": args.opset,
        "quantized": not args.no_quantize,
        "quantization": "fp16" if not args.no_quantize else None,
        "files": {
            "fp32": str(fp32_path),
            "fp32_sim": str(simplified_path),
            "fp16": str(fp16_path) if not args.no_quantize else None,
            "final": str(final_path),
        },
        "sizes_kb": {
            "fp32": fp32_path.stat().st_size / 1024,
            "fp32_sim": simplified_path.stat().st_size / 1024,
            "fp16": fp16_path.stat().st_size / 1024 if not args.no_quantize else None,
            "final": final_path.stat().st_size / 1024,
        },
    }

    if args.validate_data is not None:
        print(f"[export] evaluating fp32 simplified on '{args.validate_split}'...",
              file=sys.stderr)
        fp32_metrics = evaluate_onnx_cer(simplified_path, args.validate_data,
                                         args.validate_split)
        report["fp32_metrics"] = fp32_metrics
        print(f"  fp32: CER={fp32_metrics['cer']:.4f}  EM={fp32_metrics['em']:.3f}",
              file=sys.stderr)
        if not args.no_quantize:
            print(f"[export] evaluating fp16 on '{args.validate_split}'...",
                  file=sys.stderr)
            fp16_metrics = evaluate_onnx_cer(fp16_path, args.validate_data,
                                             args.validate_split)
            report["fp16_metrics"] = fp16_metrics
            print(f"  fp16: CER={fp16_metrics['cer']:.4f}  EM={fp16_metrics['em']:.3f}",
                  file=sys.stderr)
            cer_delta = fp16_metrics["cer"] - fp32_metrics["cer"]
            report["fp16_vs_fp32_cer_delta"] = cer_delta
            print(f"  Δ CER (fp16 - fp32) = {cer_delta:+.4f}", file=sys.stderr)
            if cer_delta > 0.002:
                print(f"  WARNING: fp16 degraded CER by {cer_delta:+.4f} "
                      f"(> 0.002 threshold per HANDOFF Phase 2 DoD)", file=sys.stderr)

    report_path = args.output_dir / f"{args.name}.report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[export] report written: {report_path}", file=sys.stderr)
    print(f"[export] final model: {final_path} "
          f"({report['sizes_kb']['final']:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
