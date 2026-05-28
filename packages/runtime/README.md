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

`MeibanOCR.recognize()` は内部で **「全体画像 → 候補テキスト行 bbox 列挙 → CRNN 認識 → 補正」**
の順に処理する。最初のステップ "候補 bbox 列挙" が **detector** で、3 つの選択肢がある。

### (1) Sliding-window (default、ゼロ依存)

```ts
const ocr = await MeibanOCR.create();  // 省略 = sliding-window
// or 明示的に
import { createSlidingWindowDetector } from '@meiban-ocr/runtime';
const ocr = await MeibanOCR.create({
  detector: createSlidingWindowDetector({ strideX: 32, strideY: 16 }),
});
```

総当たり方式。Reticle 切り出し済の小入力 (e.g. 600x200) に最適、全体フレームには遅い。

### (2) OpenCV.js 古典検出 (v0.2.0 新規、推奨)

```ts
import { MeibanOCR } from '@meiban-ocr/runtime';
import { createOpenCvDetector } from '@meiban-ocr/runtime/detectors/opencv';
// 利用側が opencv.js を持ち込む (~10MB、本パッケージはバンドルしない)
import cv from '@techstark/opencv-js';

const ocr = await MeibanOCR.create({
  detector: createOpenCvDetector(cv),
});
```

adaptive threshold + morphology + contours で text 行を絞り込み、CRNN を 5〜30 回だけ呼ぶ。
sliding-window より **20-50倍速**、銘板のような高コントラスト対象では精度も同等以上。

#### npm 経由 ではなく CDN 経由でロード (v0.2.2+)

bundler 設定を触りたくない場合は `loadOpenCv()` で CDN から動的ロードできる:

```ts
import { MeibanOCR } from '@meiban-ocr/runtime';
import { createOpenCvDetector, loadOpenCv } from '@meiban-ocr/runtime/detectors/opencv';

// CDN から自動ロード + ready 待ち、再呼び出しは window.cv を再利用
const cv = await loadOpenCv();
const ocr = await MeibanOCR.create({ detector: createOpenCvDetector(cv) });
```

メリット:
- npm に `@techstark/opencv-js` 等を入れる必要なし
- Turbopack / Webpack 設定一切不要
- ブラウザキャッシュが効くので初回以外は高速

デメリット:
- 初回 ~9MB ダウンロード (オフライン NG)
- CDN 依存 (外部サービス)

オプション:
```ts
await loadOpenCv({
  cdnUrl: 'https://docs.opencv.org/4.10.0/opencv.js',  // default
  timeoutMs: 30_000,  // default
  useExisting: true,  // default; window.cv あれば再利用
});
```

#### Next.js + Turbopack で `fs / path / crypto` 解決エラーになる場合

`@techstark/opencv-js` は Node 組込モジュールを静的 import するため、Turbopack の
client bundle で `fs can't be resolved` 等が出ることがある。`next.config.{ts,js,mjs}`
に下記を追加すると解消:

```ts
const nextConfig: NextConfig = {
  // ...既存設定...
  turbopack: {
    resolveAlias: {
      fs:     { browser: "data:text/javascript,export default {}" },
      path:   { browser: "data:text/javascript,export default {}" },
      crypto: { browser: "data:text/javascript,export default {}" },
    },
  },
  webpack: (config) => {
    // next build が webpack を使う環境の保険
    config.resolve = config.resolve || {};
    config.resolve.fallback = {
      ...(config.resolve.fallback ?? {}),
      fs: false, path: false, crypto: false,
    };
    return config;
  },
  transpilePackages: ["@techstark/opencv-js"],
};
```

確認済: Next.js 16 + Turbopack で動作。

### (3) 独自検出器 (学習済モデル / Reticle 固定 / etc)

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

## v0.2.0 の制限

- **Single product family**: 訓練データは Ericsson 4 製品 (RRU 22F3, RRUS 11 B1, Radio 2218 B42B, Radio 2251 B18 B280) のみ。未学習銘板では精度低下の可能性あり。val_CER 3.85%, val_EM 53.8% (v0 ベンチマーク)。
- **Model size**: 3 MB FP32 (FP16/INT8 化は次バージョンで検討)。
- **OpenCV detector の場合 opencv.js が peer-dependency 扱い** (本パッケージはバンドルしない、~10MB の利用側ロード必要)。

## Performance reference (HANDOFF.md 目標値)

1440×1080 nameplate sheet、20 ラベル想定:

| EP | 1 フレーム合計 | 1 ラベルあたり |
|---|---|---|
| WebGPU | ~120 ms | ~6 ms |
| WASM (4 threads) | ~600 ms | ~30 ms |

実測は環境依存。

## License

Apache-2.0. Model artifact ships under the same license as part of this package.
