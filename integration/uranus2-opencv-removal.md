# uranus2 用: OpenCV.js 撤去 + meiban-ocr v0.3.0 移行手順

`@meiban-ocr/runtime@0.3.0` で OpenCV.js helper が削除されたため、uranus2 側を sliding-window 路線に切り替える。

## なぜ撤去するか

- OpenCV.js (`@techstark/opencv-js`) は ~10MB の bundle 肥大化要因
- フロントエンドの初回ロードを大幅に遅延させる
- Reticle (~600x200 px) 用途では sliding-window で十分速度・精度確保可能
  - 窓数 ~165、WebGPU バッチ推論で 200-500ms/frame
- 検出器が必要な複雑ケースは利用側で `DetectorFn` を自前実装できる

## 撤去手順

### 1. npm 依存削除

```bash
cd ~/jdf-dev/uranus2/client
npm uninstall @techstark/opencv-js
# package-lock.json も更新されることを確認
```

### 2. `next.config.ts` の Turbopack/Webpack パッチを revert

以前 (uranus2-opencv-patch.diff) 適用した内容を逆向きに削除:

```diff
 const nextConfig: NextConfig = {
   // ...既存設定...

-  // Why: @techstark/opencv-js は Node 組込 (fs/path/crypto) を静的に
-  // import するため、Turbopack の client bundle で解決エラーになる。
-  // 空 ESM モジュール (data: URL) に alias して回避する。
-  // また、Next の compilation を強制適用するため transpilePackages にも入れる。
-  turbopack: {
-    resolveAlias: {
-      fs:     { browser: "data:text/javascript,export default {}" },
-      path:   { browser: "data:text/javascript,export default {}" },
-      crypto: { browser: "data:text/javascript,export default {}" },
-    },
-  },
-  webpack: (config) => {
-    config.resolve = config.resolve || {};
-    config.resolve.fallback = {
-      ...(config.resolve.fallback ?? {}),
-      fs: false, path: false, crypto: false,
-    };
-    return config;
-  },
-  transpilePackages: ["@techstark/opencv-js"],

   experimental: { ... },
 };
```

### 3. `meiban-ocr/runtime` を v0.3.0 へ上げる

```bash
npm install @meiban-ocr/runtime@latest
# package.json で "@meiban-ocr/runtime": "^0.3.0" になっていることを確認
```

### 4. アダプタ層からの opencv 参照削除

`useOcrProvider` 抽象を導入していれば、Meiban provider 部分を以下のように調整:

```ts
// Before (v0.2.x 系)
import { MeibanOCR } from '@meiban-ocr/runtime';
import { createOpenCvDetector, loadOpenCv } from '@meiban-ocr/runtime/detectors/opencv';

export async function createMeibanProvider(config = {}) {
  const cv = await loadOpenCv();
  const ocr = await MeibanOCR.create({
    vendor: 'ericsson',
    detector: createOpenCvDetector(cv),
  });
  return { /* ... */ };
}
```

```ts
// After (v0.3.0)
import { MeibanOCR } from '@meiban-ocr/runtime';

export async function createMeibanProvider(config = {}) {
  const ocr = await MeibanOCR.create({
    vendor: 'ericsson',
    // detector 省略 = 組込 sliding-window が動く
    // Reticle 入力チューニング (お好み):
    detector: { strideX: 24, strideY: 12 },
    minConfidence: 0.5,
  });
  return { /* ... */ };
}
```

### 5. 動作確認

```bash
# 開発サーバ
npm run dev
# Chrome DevTools の Network タブで:
# - opencv* が無くなっている
# - 初回ロードの transferred 合計が大幅減 (10MB 以上削減見込み)
# - MeibanOCR.create() → recognize() がエラー無く実行
```

### 6. 期待される効果

| 指標 | v0.2.x + OpenCV | v0.3.0 sliding-window |
|---|---|---|
| 初回 page load (bundle 同梱時) | ~13-20 MB | **~3-10 MB** |
| 初回 page load (CDN 経由 OpenCV) | ~13 MB (recognize 直前 +9 MB) | **~3-10 MB** (追加 0) |
| recognize() 速度 (Reticle 600x200) | 50-200 ms (検出後 5-10 窓) | 200-500 ms (165 窓 WebGPU バッチ) |
| 1ラベル精度 | 同等 | 同等 |
| 依存数 | +1 (`@techstark/opencv-js`) | -1 (削除) |

**トレードオフ**: recognize() あたり数百ms 遅くなる代わりに、初回 page load が**大幅に**速くなる。Reticle ユースケースなら大半の場合 sliding-window で十分。

### 7. ロールバック条件

もし sliding-window では精度・速度が足りないと判明したら:
- OpenCV.js を再導入する場合、利用側で `<script>` タグ等で OpenCV をロード後、独自 `DetectorFn` を実装する形になる
- v0.2.3 の `createOpenCvDetector` / `loadOpenCv` は npm から消えていないので戻すことも可能 (`npm install @meiban-ocr/runtime@0.2.3`)

---

## 推奨適用フロー

1. 別ブランチ (例: `chore/drop-opencv`) を切る
2. 上記 1-4 を実施、コミット
3. dev サーバで動作確認 (Network タブ、recognize 速度・精度)
4. PR → review → merge
5. 本番デプロイで初回 page load 体感比較
