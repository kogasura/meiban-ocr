# Security Policy

`@meiban-ocr/runtime` and the `meiban-ocr` repository's security policy.

## Supported versions

| Version | Status |
|---|---|
| **0.3.1** | **active (recommended)** |
| 0.3.0 | deprecated — missing `modelUrl` scheme validation |
| ≤ 0.2.3 | deprecated — missing `cdnUrl` scheme validation (v0.2.0-0.2.2) or missing `modelUrl` validation (v0.2.3) |

Use the **latest** version. `npm install @meiban-ocr/runtime@latest`.

## Reporting a vulnerability

Please **do not file a public GitHub issue** for security vulnerabilities.

### Preferred channel

GitHub **Private Vulnerability Reporting**:

1. Open <https://github.com/kogasura/meiban-ocr/security/advisories/new>
2. Fill in the vulnerability details
3. Submit (private to repo maintainers only)

We aim to respond within **5 business days**.

### Alternative

If GitHub Private Vulnerability Reporting is unavailable, contact the maintainer:

- GitHub: [@kogasura](https://github.com/kogasura)
- Open a draft issue marked `[SECURITY - DO NOT PUBLISH DETAILS]` and we will redirect

## Scope

In scope:
- `@meiban-ocr/runtime` (npm package, this repo)
- ONNX models distributed via npm
- `packages/trainer/` Python training code (locally executable scripts)
- Integration patterns documented in `integration/`

Out of scope:
- `onnxruntime-web` internal vulnerabilities (report to <https://github.com/microsoft/onnxruntime>)
- WebAssembly / WebGPU implementation issues in browsers
- Vulnerabilities in consumer applications using this library (responsibility lies with the integrator)

## What we treat as a vulnerability

- Code execution paths reachable from `recognize()` / `MeibanOCR.create()` inputs (XSS, RCE, prototype pollution)
- Memorized real serial leakage via the bundled ONNX model
- Information disclosure (real serial numbers, internal data, credentials)
- Supply chain compromise of the published package
- Path traversal / file access in `modelUrl` / `cdnUrl` parameters
- Authentication bypass for npm publish (organization scope hijack)

## Known limitations (intentional design)

- **`OCRResult.text` is untrusted output.** It is the recognized text from
  user-supplied images, which may be camera-captured banners, labels, etc.
  Consumers MUST NOT pass it directly into `innerHTML`, SQL queries, system shell, etc.
  Always treat it as user input. See `packages/runtime/README.md` "Security considerations".
- **`modelUrl` accepts any `http://` / `https://` host.** Cross-origin URLs work
  but consumers should only point at trusted CDNs they control.
- **Bundled ONNX model is not signed.** In-package data URL inline cannot use SRI.
  Future versions (v0.4.0+) will move toward external `.onnx` + SHA-256 verification.

## Training data and model behavior

### Training data is not redistributed

Training data (photographic samples of Ericsson nameplates) is kept **private**:

- `samples/`, `samples_test/`, `videos/`, `data/recognition/{train,val,test}/`
  → all gitignored (never committed)
- `runs/` (training artifacts including model predictions) → gitignored (since v0.3.x)
- `annotations/*.json` → committed but with **synthetic dummy serial labels** only
  (real serial numbers replaced with `E300MM000XXX` / `E300MM999XXX` synthetic dummies)
- The bundled ONNX model is included in the npm package, but the training data itself
  is **not part of the distribution**.

### Model behavior is standard OCR

The bundled CRNN model is a general OCR model trained to recognize 36-character
alphanumeric text. Like any OCR model:

- Given an image containing readable text, it outputs that text.
- This is true whether the input text happens to be a real Ericsson serial, a
  synthetic dummy, or arbitrary alphanumeric content.
- This is not "memorization leakage" — it is the model's intended function.

### Why we did not retrain on synthetic-only data

We considered retraining the model exclusively on synthetic dummies to eliminate
theoretical membership-inference attacks. After review we concluded:

- The training data files are already private (gitignored), so the strongest leak
  vector is closed.
- Membership inference on a 38-real-sample CRNN provides minimal practical
  information (the search space is too small to be useful).
- Other widely-used OCR libraries (Tesseract, PaddleOCR, RapidOCR, etc.) are
  trained on real text and ship freely; this is the industry norm.

If you have a stricter NDA requirement where the bundled model itself must not have
been exposed to real customer data, we recommend `MeibanOCR.create({ modelUrl })`
pointing at your own internally-trained model.

### Sanitization details (for reproducibility)

If you discover a synthetic dummy value (`E300MM...`) that collides with a real
production serial unknown to us, please report it via the vulnerability channel
above so we can switch the dummy range.

## License

Apache-2.0. Reporting a vulnerability does not affect the license of any submitted
proof-of-concept code — we assume reasonable good-faith disclosure.
