# meiban-ocr-trainer

Python training side for `meiban-ocr`. See `../../HANDOFF.md` for the full spec.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,export,metrics]"
pytest
```

`torch` is heavy; CPU-only install is fine for skeleton tests.

## Layout

```
src/meiban_ocr_trainer/
├── constants.py            # CHARSET / NUM_CLASSES / BLANK_IDX (shared with runtime)
├── tokenizer.py            # CTC tokenizer (encode/decode + greedy)
├── models/
│   └── tiny_ocr.py         # CRNN (MobileNetV3-Small + Bi-GRU + CTC)
├── data/
│   ├── extract_frames.py   # video → frames (Phase 1, Day 2)
│   ├── extract_crops.py    # Stage 1 → Stage 2 (Day 3)
│   ├── text_replace.py     # text-rewrite augmentation (Day 3)
│   ├── refine_bbox.py      # OpenCV bbox refinement (Day 3)
│   ├── dataset.py          # PyTorch Dataset / DataLoader (Day 4)
│   └── augment.py          # Albumentations pipeline (Day 4)
├── train.py                # config-driven training entry (Day 6)
├── evaluate.py             # CER / WER / EM (Day 7)
└── export.py               # PyTorch → ONNX + INT8 quantization (Phase 2)
```

Each `data/*.py` module is a Day 1 stub with a NotImplementedError-style body
plus a docstring describing the contract. They get filled in during Phase 1.

## Scripts

```bash
meiban-ocr-train  --config configs/default.yaml
meiban-ocr-eval   --checkpoint runs/latest/best.pt --data data/recognition/test
meiban-ocr-export --checkpoint runs/latest/best.pt --output models/tiny-ocr-v1.onnx
```

(All three currently print a "not implemented in Day 1" message.)
