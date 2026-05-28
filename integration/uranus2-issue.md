# [Mobile/NameplateStocktake] OCR エンジン抽象化 + @meiban-ocr/runtime 試験導入

## 背景

現状の `useNameplateOcr.ts` は `tesseract.js@5.1.x` (eng_best, LSTM) に直接依存している。
URANUS2 OCR エンジン比較レポート ([`~/jdf-dev/output/uranus2/ocr-engine-comparison-report.md`](./output/uranus2/ocr-engine-comparison-report.md)) で、Tesseract は
本対象 (`E[39]\d{2}MM\d{6}`) に対し coverage 60% (eng_best + crop) が限界と判明。

並行して銘板 OCR 専用に CRNN を訓練 + ONNX 化した `@meiban-ocr/runtime` を準備済み
(別レポ: `~/meiban-ocr/`、v0.1.0 はバンドル ONNX 3MB)。

URANUS2 で **両方を切替可能**にし、A/B 比較 + 段階的移行できる構造に再設計したい。

---

## ゴール

1. OCR エンジン部を抽象化し、`tesseract` / `meiban` をランタイム切替可能にする
2. 既存 UI の挙動を壊さない (confidence しきい値 / OCR_INTERVAL_MS / matchOcr 互換維持)
3. デバッグ画面 (`debugMode`) に「現在のエンジン名」と「切替トグル」を追加
4. A/B 比較メトリクスを収集できる土台 (どちらのエンジンが何件マッチしたか)

非ゴール:
- tesseract.js を即削除 (両者並行運用を継続)
- 多ベンダー (Samsung 等) 対応 (Meiban は Ericsson 専用)

---

## 提案アーキテクチャ

```
[既存]
NameplateOcrScanner.tsx
  └ useNameplateOcr({psm, whitelist})
      └ tesseract.js (固定)

[提案]
NameplateOcrScanner.tsx
  └ useOcrProvider(engineName, config)   ← 切替可能
      ├ createTesseractProvider({psm, whitelist})
      ├ createMeibanProvider({vendor: 'ericsson'})
      └ (将来) createRapidOcrProvider({endpoint})
```

### 共通インターフェース

```ts
// src/features/0_Mobile/MobileNameplateStocktake/Scanner/ocr/types.ts
export type OcrProviderName = 'tesseract' | 'meiban';

export interface OcrResult {
  text: string;
  confidence: number;   // 0-100 に統一 (uranus2 既存しきい値ロジック互換)
  bbox?: [number, number, number, number];  // Meiban のみ返す
}

export interface OcrProvider {
  readonly name: OcrProviderName;
  recognize(image: HTMLCanvasElement): Promise<OcrResult | null>;
  dispose(): Promise<void>;
}
```

### React フック

```ts
// src/features/0_Mobile/MobileNameplateStocktake/Scanner/ocr/useOcrProvider.ts
export function useOcrProvider(name: OcrProviderName, config?: OcrProviderConfig) {
  // 既存 useNameplateOcr と同じ shape を返す: { recognize, isReady, initError }
  // name 変更時は前 provider を dispose、新 provider を await create
}
```

`name` を `useState` 化することで UI トグルから差し替え可能。

---

## 実装計画 (段階)

### Phase A: 抽象レイヤー追加 (互換維持)

- [ ] `src/.../Scanner/ocr/types.ts` 追加
- [ ] `src/.../Scanner/ocr/tesseract-provider.ts` 追加 (現行 useNameplateOcr 中身を移植)
- [ ] `src/.../Scanner/ocr/useOcrProvider.ts` 追加
- [ ] `useNameplateOcr.ts` を `useOcrProvider('tesseract', {psm, whitelist})` の thin wrapper に変更
- [ ] 既存 NameplateOcrScanner.tsx は無変更で動くことを確認
- [ ] テスト: 既存挙動と diff なし

参考実装: <https://github.com/meiban-ocr/meiban-ocr/tree/main/packages/runtime/examples/react-tesseract-compat>

### Phase B: @meiban-ocr/runtime プロバイダ追加

- [ ] `npm i @meiban-ocr/runtime onnxruntime-web` (client/ に追加)
- [ ] `src/.../Scanner/ocr/meiban-provider.ts` 追加
  - `MeibanOCR.create({vendor: 'ericsson'})` で初期化
  - `recognize` は `OCRResult[]` の最 confidence 1件を返す (Tesseract 形式に整形)
  - `confidence *= 100` でスケール統一
- [ ] `useOcrProvider` の switch に `'meiban'` を追加
- [ ] UI トグル追加 (debugMode のみ表示)
- [ ] テスト: meiban エンジンでも recognize → matchOcr のループが回る

### Phase C: A/B 比較 + メトリクス

- [ ] エンジン名を `useScanQueue` のエントリに記録 (`ocr_engine` フィールド)
- [ ] 同一画像で両エンジンを並列実行する "shadow mode" (デバッグオプション)
- [ ] CER 計測スクリプト (期待 serial リスト vs 検出結果)
- [ ] Issue でレポート公開

### Phase D: 既定切替 (将来)

- [ ] meiban の性能/精度が tesseract を全面的に上回ったら、production デフォルトを切替
- [ ] tesseract は fallback として保持

---

## API シェイプの確認

### Before (現行)

```tsx
const { recognize, isReady, initError } = useNameplateOcr({ psm, whitelist });
// recognize: (img: ImageLike) => Promise<{text, confidence} | null>
```

### After (提案、最小変更)

```tsx
const { recognize, isReady, initError } = useOcrProvider('tesseract', { psm, whitelist });
// 同じ shape (recognize, isReady, initError) で互換
```

### A/B 切替時

```tsx
const [engine, setEngine] = useState<OcrProviderName>('tesseract');
const ocr = useOcrProvider(engine, engineConfig(engine));

// UI:
<ToggleButtonGroup value={engine} onChange={(_, v) => v && setEngine(v)}>
  <ToggleButton value="tesseract">Tesseract</ToggleButton>
  <ToggleButton value="meiban">MeibanOCR</ToggleButton>
</ToggleButtonGroup>
```

---

## 影響範囲 + 変更ファイル

- 新規: `client/src/features/0_Mobile/MobileNameplateStocktake/Scanner/ocr/{types,useOcrProvider,tesseract-provider,meiban-provider}.ts`
- 改修: `useNameplateOcr.ts` (thin wrapper化)
- 改修: `NameplateOcrScanner.tsx` (Phase B 以降、エンジントグル UI)
- 改修: `client/package.json` (deps 追加: `@meiban-ocr/runtime`, `onnxruntime-web`)
- 変更不要: `useScanQueue.ts`, `matchOcr`, `vendorOcrPatterns.ts`, `index.tsx`

`MobileScannerBench` (`runBench.ts`, `BenchControls.tsx`, `index.tsx`) も同様に
provider 抽象を経由する形で書き換え可能 (Phase B と並行)。

---

## 配信サイズへの影響 (Phase B)

- `@meiban-ocr/runtime`: 3 MB (ONNX 含む、ESM)
- `onnxruntime-web` (peer): ~10-15 MB (wasm/webgpu バックエンド、初回フェッチ)
- Tesseract.js eng_best traineddata: 22 MB (CDN から初回 fetch)

Tesseract 既存初回コスト (22MB DL) より重くなる可能性は低い。
詳細は実機ベンチで要確認。

---

## リスクとロールバック

| リスク | 対策 |
|---|---|
| @meiban-ocr/runtime が本番ブラウザで動かない | feature flag (env var) で tesseract に強制 fallback |
| confidence スケール変換ミス | adapter テスト + shadow mode で比較 |
| onnxruntime-web の WASM ロード失敗 | EP fallback (webgpu → wasm)、最悪 throw → tesseract に自動退避 |
| ライブラリ依存追加 (license, supply chain) | Apache-2.0、HF Hub 公開モデル、ライセンス監査 OK |

ロールバック: provider 名のデフォルトを `tesseract` 固定にすれば即時。

---

## 関連

- `@meiban-ocr/runtime` リポジトリ: `~/meiban-ocr/`
- 訓練データ + 結果: `~/meiban-ocr/runs/20260527-192215/`
  - val CER 3.85%、val EM 53.8% (img_002 Radio 2218 B42B 13ラベル)
- OCR エンジン比較レポート: `output/uranus2/ocr-engine-comparison-report.md`
- HANDOFF.md / HANDOFF_ADDENDUM.md (meiban-ocr 仕様)

---

## 受入基準

- [ ] Phase A 完了: 既存挙動と pixel-level 一致
- [ ] Phase B 完了: debugMode で `engine='meiban'` に切替えると recognize が動作
- [ ] Phase C 完了: A/B レポートが 1 件公開される
- [ ] Phase D 開始は別 issue で議論
