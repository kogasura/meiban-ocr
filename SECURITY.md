# Security Policy

`@meiban-ocr/runtime` and the `meiban-ocr` repository's security policy.

## Supported versions

| Version | Status |
|---|---|
| 0.4.x | not yet released |
| 0.3.1 | **deprecated** — bundled ONNX model may contain memorized training data |
| 0.3.0 | **deprecated** — missing `modelUrl` scheme validation |
| ≤ 0.2.3 | **deprecated** — missing `cdnUrl` scheme validation (v0.2.0-0.2.2) or missing `modelUrl` validation (v0.2.3) |

Use the **latest non-deprecated** version. `npm install @meiban-ocr/runtime@latest`.

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

## Training data disclosure

Training data was sourced from photographic samples of Ericsson nameplates. All real
serial numbers have been sanitized from the public repository and replaced with
synthetic dummies (prefix `E300MM`, range `000XXX` and `999XXX`) that are intentionally
chosen to be **structurally similar but not in any real Ericsson production range
we are aware of**. Should you identify a synthetic value that happens to collide with a
real production serial, please report it via the vulnerability channel above.

## License

Apache-2.0. Reporting a vulnerability does not affect the license of any submitted
proof-of-concept code — we assume reasonable good-faith disclosure.
