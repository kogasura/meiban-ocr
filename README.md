# meiban-ocr

[![CI](https://github.com/kogasura/meiban-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/kogasura/meiban-ocr/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/@meiban-ocr/runtime.svg)](https://www.npmjs.com/package/@meiban-ocr/runtime)
[![npm downloads](https://img.shields.io/npm/dm/@meiban-ocr/runtime.svg)](https://www.npmjs.com/package/@meiban-ocr/runtime)
[![license](https://img.shields.io/npm/l/@meiban-ocr/runtime.svg)](./LICENSE)

Browser-friendly OCR for industrial nameplate serial codes
(primary target: Ericsson `E300MM000032` format).

PyTorch (CRNN: MobileNetV3-Small + Bi-GRU + CTC) for training,
`onnxruntime-web` (WebGPU + WASM) for inference.

> Status: **v0.2.1 published on npm**. Use it via `npm i @meiban-ocr/runtime onnxruntime-web`.
> See [`packages/runtime/README.md`](./packages/runtime/README.md) for the API.

## Links

- npm: <https://www.npmjs.com/package/@meiban-ocr/runtime>
- GitHub: <https://github.com/kogasura/meiban-ocr>
- Runtime API doc: [`packages/runtime/README.md`](./packages/runtime/README.md)
- Full spec: [`HANDOFF.md`](./HANDOFF.md) + [`HANDOFF_ADDENDUM.md`](./HANDOFF_ADDENDUM.md)
- uranus2 integration plan: [`integration/uranus2-issue.md`](./integration/uranus2-issue.md)

## Repository layout

```
meiban-ocr/
├── HANDOFF.md / HANDOFF_ADDENDUM.md  # Full spec
├── CLAUDE.md / LABELING.md           # Working context
├── packages/
│   ├── trainer/       # Python training side (PyTorch + ONNX export)
│   └── runtime/       # TypeScript inference side, npm-publishable as @meiban-ocr/runtime
├── annotations/       # Stage 1 JSON annotations (4 images, 54 labels, claude-verified)
├── runs/              # Training run artifacts (summary.json, val_preds.json)
├── integration/       # uranus2 issue body + npm publish guide
├── samples/, samples_test/, videos/, data/, fonts/, models/  # gitignored
```

## Quick start

### Use the published npm package

```bash
npm i @meiban-ocr/runtime onnxruntime-web
```

```ts
import { MeibanOCR } from '@meiban-ocr/runtime';
const ocr = await MeibanOCR.create({ vendor: 'ericsson' });
const results = await ocr.recognize(canvas);
// → [{ text: 'E300MM000032', confidence: 0.96, bbox: [x1,y1,x2,y2] }, ...]
await ocr.dispose();
```

### Develop locally

```bash
# JavaScript side
pnpm install
pnpm -F @meiban-ocr/runtime test
pnpm -F @meiban-ocr/runtime build

# Python side
cd packages/trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,export]"
pytest
```

WSL Ubuntu で `python3 -m venv` が動かない場合は `pip3 install --user pytest torch torchvision albumentations torchmetrics` で代用 → `PYTHONPATH=src pytest` で実行。

## Current model performance (v0.2.1)

| 指標 | 値 |
|---|---|
| val CER (img_002, 13 labels) | **3.85%** |
| val EM (完全一致率) | **53.8%** (7/13) |
| ONNX size (FP32) | **3.0 MB** |
| 1ラベル推論 (WebGPU 想定) | < 20 ms |

訓練データ: Ericsson 4 製品 (RRU 22F3, RRUS 11 B1, Radio 2218 B42B, Radio 2251 B18 B280)
4 枚 / 54 ラベル (RapidOCR 自動ラベル + Claude VLM ダブルチェック)。

## Roadmap (要約)

| Phase | Status | 内容 |
|---|---|---|
| 1 | ✅ done | データパイプライン + CRNN訓練 (val_CER 3.85%) |
| 2 | ✅ done | ONNX export + onnx-simplifier (FP32 3MB) |
| 3 | ✅ done | TypeScript ランタイム + npm publish (v0.2.1) |
| 1.5 | ⏸ pending | 動画フレームから data scale → 再訓練 |
| 4 | ⏸ pending | Fine-tune + HuggingFace Hub publish + 多ベンダー |

## License

Apache 2.0. See [`LICENSE`](./LICENSE).
