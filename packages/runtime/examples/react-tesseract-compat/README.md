# react-tesseract-compat: uranus2 用 OCR Provider 抽象アダプタ

uranus2 の既存 `useNameplateOcr` (Tesseract.js 専用) を、**OCR エンジン切替可能**な
抽象レイヤーに置き換えるための参考実装。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `types.ts` | `OcrProvider` インターフェース、`OcrResult`、共通型 |
| `meiban-provider.ts` | `@meiban-ocr/runtime` を OcrProvider に適合 (今回追加) |
| `tesseract-provider.ts` | 現行 tesseract.js を OcrProvider に適合 (リフレーム) |
| `useOcrProvider.ts` | `name='tesseract' \| 'meiban'` で動的切替する React フック |

## 統合の最小差分 (uranus2 側で必要な変更)

### Before

```tsx
import { useNameplateOcr } from './useNameplateOcr';
const { recognize, isReady, initError } = useNameplateOcr({ psm, whitelist });
```

### After

```tsx
import { useOcrProvider } from './ocr/useOcrProvider';
const [engine, setEngine] = useState<'tesseract' | 'meiban'>('tesseract');
const { recognize, isReady, initError, activeName } = useOcrProvider(engine, {
  psm,
  whitelist,
});
// UI トグルで setEngine('meiban') すれば実行時切替できる
```

呼出側 (`NameplateOcrScanner.tsx` の `recognize(canvas)` ループ) の変更は不要。
返り値の `text` / `confidence` は同じ shape (confidence は 0-100 にスケール済)。

## なぜ抽象化するか

1. **A/B 比較しやすい**: 同じ画像を tesseract と meiban で同時に走らせ、CER 差を実測
2. **段階的移行**: 一気に置き換えず、ユーザー単位で feature flag で切替
3. **将来追加**: rapidocr (サーバ), parseq などを足すときに provider 追加するだけ
4. **ロールバック容易**: 不具合時にトグルで戻せる

## confidence スケールの統一

| エンジン | 元のスケール | アダプタ後 |
|---|---|---|
| Tesseract.js | 0–100 | 0–100 (そのまま) |
| @meiban-ocr/runtime | 0–1 | 0–100 (×100) |

uranus2 の `OCR_CONFIDENCE_INSTANT` 等のしきい値ロジックを変えずに済む。

## 制限事項

- `meiban` プロバイダは Ericsson 専用 (`E[39]\d{2}MM\d{6}`)。他ベンダーは tesseract 経由のまま。
- `meiban` の text は既に 6 段補正済 → uranus2 側の `matchOcr` は stage 1 通過するだけ
  (二重補正だが、結果は変わらない。次フェーズで matchOcr スキップ最適化可)。
- `meiban` は ONNX (3MB) を初期ロード時にバンドル展開 → 初回 init が tesseract より少し速い
  (CDN ダウンロード不要)、ただし JS バンドルサイズは +3MB。

## 動作確認 (optional)

例コードは本パッケージの typecheck には含まれない (`react` / `tesseract.js` 依存を本体に
持ち込まないため)。利用側に当てて検証する場合:

```bash
cd packages/runtime/examples/react-tesseract-compat
pnpm i react @types/react tesseract.js @meiban-ocr/runtime
npx tsc --noEmit  # tsconfig.json は同梱
```

## ライセンス

参考実装。uranus2 リポジトリにコピーして使用してOK (Apache-2.0)。
