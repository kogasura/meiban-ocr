# meiban-ocr

Browser-friendly, lightweight alphanumeric OCR for industrial nameplate serial codes
(primary target: Ericsson `E300MM000032` format).

PyTorch (CRNN: MobileNetV3-Small + Bi-GRU + CTC) for training, `onnxruntime-web`
(WebGPU + WASM) for inference.

> Status: **Phase 1 setup (Day 1 skeleton)**. No model artifact yet — see
> `HANDOFF.md` for the full roadmap.

## Repository layout

```
meiban-ocr/
├── HANDOFF.md         # Full spec (read first)
├── HANDOFF_ADDENDUM.md # Addendum #1: pretrained backbone + Plan B/C/D
├── CLAUDE.md          # Short context for Claude Code
├── LABELING.md        # Labeling instructions for samples/
├── packages/
│   ├── trainer/       # Python training side (PyTorch + ONNX export)
│   └── runtime/       # TypeScript inference side (npm-publishable)
├── videos/            # Raw videos (gitignored)
├── samples/           # Extracted frames + photos (gitignored)
├── samples_test/      # Held-out test images (isolated, never labeled by Claude)
├── annotations/       # Stage 1 JSON annotations (git-managed)
├── data/recognition/  # Stage 2 crops for training (gitignored)
├── fonts/             # OCR-A / OCR-B for text-replace augmentation
└── models/            # ONNX artifacts (Git LFS or HF Hub)
```

## Quick start

```bash
# JavaScript side
pnpm install
pnpm -F @meiban-ocr/runtime test

# Python side
cd packages/trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Phase 1 prerequisites (user must provide)

- 5–10 short videos (~1 min each) of nameplates
- 30–50 direct photos
- 50 held-out test images in `samples_test/` (never used for training)
- OCR-A / OCR-B fonts in `fonts/` (Open Font Licensed)

See `HANDOFF.md §11` for the full checklist.

## License

Apache 2.0. See `LICENSE`.
