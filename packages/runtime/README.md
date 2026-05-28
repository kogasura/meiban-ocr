# @meiban-ocr/runtime

Browser-friendly OCR for industrial nameplate serials (Ericsson `E300MM000032` format).
ONNX CRNN model bundled in-package, no separate model file or CDN required.

- Inference engine: `onnxruntime-web` (WebGPU → WASM fallback)
- Model: MobileNetV3-Small + Bi-GRU + CTC, ~3 MB FP32 ONNX (bundled inline)
- Detection: built-in sliding-window proposal generator (no OpenCV.js dependency)
- License: Apache-2.0

## Install

```bash
npm i @meiban-ocr/runtime onnxruntime-web
# or pnpm / yarn
```

`onnxruntime-web` is a peer-style dependency — you bring your own version.

## Usage

```ts
import { MeibanOCR } from '@meiban-ocr/runtime';

// init: モデルはバンドル済み、追加ダウンロード不要
const ocr = await MeibanOCR.create({ vendor: 'ericsson' });

// camera frame / Reticle crop / 任意の Canvas 画像を渡せる
const results = await ocr.recognize(canvas);
// → [{ text: 'E300MM000032', confidence: 0.96, bbox: [120, 340, 480, 420] }, ...]
// 該当なし → []

await ocr.dispose();
```

### API

```ts
type ImageInput = HTMLCanvasElement | OffscreenCanvas | ImageBitmap | ImageData;

interface MeibanOCROptions {
  vendor?: 'ericsson';                                  // 補正パイプライン (default)
  executionProviders?: Array<'webgpu' | 'wasm'>;         // ORT EP (default: ['webgpu', 'wasm'])
  minConfidence?: number;                                // default 0.5
  modelUrl?: string;                                      // バンドル版を上書きしたいとき
  modelBytes?: Uint8Array | ArrayBuffer;                  // ↑ バイト列で渡すとき
  detector?: {
    maxInputDim?: number;                                 // default 1024
    strideX?: number;                                     // default 32
    strideY?: number;                                     // default 16
    scales?: number[];                                    // default [1.0]
  };
  maxBatchSize?: number;                                  // default 64
}

interface OCRResult {
  text: string;            // 確定値 (regex 通過済み)
  confidence: number;      // 0-1
  bbox: [number, number, number, number];  // 入力画像内座標 [x1,y1,x2,y2]
}

class MeibanOCR {
  static create(opts?: MeibanOCROptions): Promise<MeibanOCR>;
  recognize(image: ImageInput): Promise<OCRResult[]>;
  dispose(): Promise<void>;
}
```

## Vendor pattern

Currently supports **Ericsson** only:

- Strict regex: `/^E[39]\d{2}MM\d{6}$/`  例: `E300MM000032`, `E300MM999022`
- 6-stage correction pipeline (backend `PlateSerialNumber.php` と互換):
  1. Strict full match
  2. Strict full + `O → 0` fallback
  3. Lenient full (Ericsson 対象外、将来用)
  4. Lenient + `O → 0`
  5. Strict partial + `O → 0`
  6. Strict partial

`recognize()` の `text` は 6 段補正後の確定値です。

## 検出器 (Detector) の差し替え (v0.2.0+)

`MeibanOCR.recognize()` は内部で **「入力画像 → 候補テキスト行 bbox 列挙 → CRNN 認識 → 補正」**
の順に処理する。最初のステップ "候補 bbox 列挙" が **detector** で、2 つの選択肢がある。

### (1) Sliding-window (default、ゼロ依存) ★推奨

```ts
const ocr = await MeibanOCR.create();  // 省略 = sliding-window
// or 明示的にチューニング
import { createSlidingWindowDetector } from '@meiban-ocr/runtime';
const ocr = await MeibanOCR.create({
  detector: createSlidingWindowDetector({ strideX: 32, strideY: 16 }),
});
```

総当たり方式。**Reticle 切り出し済の小入力 (e.g. 600x200) 用途には最適**。
外部依存ゼロ・bundle bloat なし。WebGPU でバッチ推論されるので 100-500ms/frame 程度。

#### Reticle ユースケース (uranus2 等) のチューニング例

```ts
const ocr = await MeibanOCR.create({
  vendor: 'ericsson',
  detector: { strideX: 24, strideY: 12 },  // 細かめ stride で recall 上げる
  minConfidence: 0.5,
});
const results = await ocr.recognize(reticleCanvas);
```

### (2) 独自検出器 (学習済モデル / Reticle 固定 / OpenCV.js 自作 / etc)

```ts
import type { DetectorFn } from '@meiban-ocr/runtime';

const reticleDetector: DetectorFn = (image) => [
  // 入力全体を 1 つの候補として扱う (uranus2 Reticle 互換)
  [0, 0, image.width, image.height],
];

const ocr = await MeibanOCR.create({ detector: reticleDetector });
```

`DetectorFn = (image: ImageData) => BBox[] | Promise<BBox[]>` を満たせば何でも OK。

---

## v0.3.0 の変更点 (breaking)

- **OpenCV.js helper を削除**。`createOpenCvDetector` / `loadOpenCv` の export は廃止。
  独自に OpenCV.js を使いたい場合は `DetectorFn` を自前実装する形に統一。
- フロントエンド軽量化を最優先する方針。デフォルトの sliding-window で多くのユースケース
  (Reticle 切り出し等) は十分カバーできるため。
- 移行ガイド: `createOpenCvDetector(cv)` → `DetectorFn` で独自実装、もしくは省略して sliding-window を使う。

## v0.3.0 の制限

- **Single product family**: 訓練データは Ericsson 4 製品 (RRU 22F3, RRUS 11 B1, Radio 2218 B42B, Radio 2251 B18 B280) のみ。未学習銘板では精度低下の可能性あり。val_CER 3.85%, val_EM 53.8% (v0 ベンチマーク)。
- **Model size**: 3 MB FP32 (FP16/INT8 化は次バージョンで検討)。

## Security considerations

### `OCRResult.text` は untrusted な出力として扱うこと

`recognize()` の返却 `text` は、ユーザーがカメラ撮影した任意画像から抽出された
文字列です。攻撃者が **意図的な文字列を印字したラベルを撮影させる** ことで、
任意文字列を `OCRResult.text` 経由でアプリに注入する余地があります。

DO:
- `textContent` プロパティ / React の `{text}` 補間 (自動エスケープされる)
- SQL は parameterized query
- log では quote / escape

DON'T:
- `el.innerHTML = result.text`
- `eval`, `Function`, `setTimeout(string)` に渡す
- shell command / SQL に直接埋め込む

### `modelUrl` / `modelBytes` は信頼するソースからのみ

`MeibanOCR.create({ modelUrl })` の引数は scheme 検証 (`http:` / `https:` / `data:` /
`blob:` のみ許可) されますが、**host の whitelist は無し**です。
利用側は untrusted な値 (URL クエリパラメータ等) を直接渡さないこと:

```ts
// ❌ 危険: 攻撃者が ?model= で任意 ONNX を指定可能
const modelUrl = new URLSearchParams(location.search).get('model');
await MeibanOCR.create({ modelUrl });

// ✅ 自分で管理する CDN のみ
await MeibanOCR.create({ modelUrl: 'https://cdn.example.com/meiban/model.onnx' });

// ✅ 整合性検証して bytes で渡す (推奨)
const expectedSha = 'a1b2c3...';
const response = await fetch('https://cdn.example.com/model.onnx');
const buf = await response.arrayBuffer();
const hash = await crypto.subtle.digest('SHA-256', buf);
// hash を expectedSha と照合してから:
await MeibanOCR.create({ modelBytes: buf });
```

### ONNX モデル整合性

現状の bundled モデル (`@meiban-ocr/runtime` に inline されている data URL) は
SRI 不可です。npm パッケージ自体の `integrity` (`shasum`/`sha512`) は npm
レジストリで検証されるため、`npm install --ignore-scripts` 等の通常導入なら問題なし。
さらに強い検証が必要なら `modelBytes` 経由で消費側 SHA-256 検証を行ってください。

### 報告窓口

脆弱性を発見したら **public issue ではなく** GitHub Private Vulnerability Reporting
で報告してください: <https://github.com/kogasura/meiban-ocr/security/advisories/new>
詳細は [`SECURITY.md`](https://github.com/kogasura/meiban-ocr/blob/main/SECURITY.md) を参照。

## Performance reference (HANDOFF.md 目標値)

1440×1080 nameplate sheet、20 ラベル想定:

| EP | 1 フレーム合計 | 1 ラベルあたり |
|---|---|---|
| WebGPU | ~120 ms | ~6 ms |
| WASM (4 threads) | ~600 ms | ~30 ms |

実測は環境依存。

## License

Apache-2.0. Model artifact ships under the same license as part of this package.
