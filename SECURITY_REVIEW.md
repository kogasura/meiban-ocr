# Security Review (2026-05-28)

`@meiban-ocr/runtime@0.2.2` と repo 現状に対する 初回 自主セキュリティレビュー。

## まとめ

| カテゴリ | 件数 | 最高深刻度 |
|---|---|---|
| 既知 CVE (依存) | 2 | moderate (dev のみ) |
| ソースコードの懸念 | 1 | **要修正** (low-medium) |
| 機密データ漏洩 | 0 | — |
| CI/Publish 強度 | 0 重大 | — (改善余地あり) |

**Production 利用者への影響**: なし (vite/esbuild の脆弱性は dev のみ。コード懸念は呼び出し条件次第)。

---

## Finding 1: vite / esbuild の dev server CORS 設定 (依存脆弱性)

### 深刻度
**Moderate (devDependency のみ)**

### 詳細
- **GHSA-67mh-4wv8-2f99** (esbuild): dev server が `Access-Control-Allow-Origin: *` を返すため、開発中に悪意あるサイトを開くと localhost のソースを読み取られる
- **GHSA-4w7w-66w2-5vf9** (vite): 同様の dev server 系
- 影響を受けるバージョン: vite 5.4.21 (現状)、esbuild 0.21.5 (vite 経由)

### Production 影響
**なし** — `vite` と `esbuild` は devDependency。`npm publish` した `dist/` には含まれない。利用者の bundle にも入らない。

### 開発者への影響
- `pnpm dev` / `vitest watch` 実行中に**信頼できないサイトを別タブで開く**と、攻撃者がローカル開発サーバから source を引き出せる
- 実害: 個人のローカル開発フローでは限定的

### 推奨対応
**優先度: 低**。vite 5 系の最新 (5.4.21) でも未修正、vite 6.x 以降への移行で解消。後回しでよい。

```bash
# 将来対応
pnpm -F @meiban-ocr/runtime up vite@^6 vitest@^2
# breaking change がないか build / test 確認後 commit
```

---

## Finding 2: `loadOpenCv(cdnUrl)` の URL プロトコル未検証

### 深刻度
**Low-Medium (条件付き)**

### 詳細
`src/detectors/opencv.ts` の `loadOpenCv()` は引数の `cdnUrl` を `<script>.src` に直接代入する:

```ts
const script = document.createElement('script');
script.src = cdnUrl;  // ← 検証なし
document.head.appendChild(script);
```

`cdnUrl` が `javascript:alert(1)` のような擬似プロトコル URL や `data:text/javascript,...` だった場合、任意コード実行される。

### 攻撃シナリオ
1. アプリケーションが URL クエリ等から `cdnUrl` を取得しユーザー入力を `loadOpenCv({cdnUrl: untrustedInput})` に流す
2. 攻撃者が `?cdnUrl=javascript:fetch('...')` のような URL を被害者に踏ませる
3. 任意 JS が実行され、 cookie / token 等を持ち出される (DOM-based XSS)

### Production 影響
- **デフォルト使用** (`loadOpenCv()` 引数なし) は ハードコードの `https://docs.opencv.org/...` 固定なので **影響なし**
- **明示的に `cdnUrl` を渡す呼び出しで、その値が untrusted source 由来の場合のみ問題**

### 推奨対応
**優先度: 中**。v0.2.3 として URL スキーム検証を追加 (https / http のみ許可)。

```ts
function validateCdnUrl(url: string): void {
  let parsed: URL;
  try { parsed = new URL(url, location?.href ?? 'http://localhost/'); }
  catch { throw new Error(`loadOpenCv: invalid cdnUrl: ${url}`); }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error(`loadOpenCv: unsupported protocol "${parsed.protocol}". Only http/https allowed.`);
  }
}
```

---

## Finding 3: GitHub Actions workflow permissions 明示なし

### 深刻度
**Informational**

### 詳細
`.github/workflows/ci.yml` に明示的な `permissions:` ブロック無し。今日作った新規 repo は GitHub のデフォルトで `read-only`、安全寄りだが、明示しておく方が future-proof。

### 推奨対応
`permissions: {contents: read}` を追加。

```yaml
permissions:
  contents: read  # checkout のみ可、push等不可
```

---

## チェックOKだったもの (記録)

### コードパターン
- ✅ `eval()` / `new Function()` 使用なし
- ✅ `innerHTML` / `outerHTML` / `document.write` 使用なし
- ✅ Prototype pollution パターンなし (`__proto__` 操作、`prototype[]` 動的代入なし)
- ✅ `fetch` / `XMLHttpRequest` 直接呼出なし (ORT が内部で処理)

### データ漏洩
- ✅ ハードコードされた API key / secret / token / password なし
- ✅ ソースコード内に email アドレスなし (package.json のみ意図的)
- ✅ ONNX バイナリ内に絶対パス / ユーザー名なし (PyTorch 標準出力で metadata 最小限)
- ✅ Git history に commit されたシークレット なし

### Supply chain
- ✅ 依存パッケージ: production は `onnxruntime-web` 1 個のみ (Microsoft 公式)
- ✅ npm 公開時 2FA (パスキー) 必須
- ✅ npm scope `@meiban-ocr` は yuuki-okubo 所有 (taking risk なし)
- ✅ scoped publish access `public` 明示済 (誤って private 課金されない)

### 配信物
- ✅ 公開 `dist/index.js` には eval / Function / innerHTML 文字列なし
- ✅ npm tarball に余計なファイル混入なし (`.npmignore` 相当を `files` で whitelist)
- ✅ `.env` / `.npmrc` / 鍵ファイル等のコミットなし

### ライセンス
- ✅ Apache-2.0 (LICENSE 同梱)
- ✅ ONNX モデルも同ライセンス下で配布 (README に明記)
- ✅ dependency: `onnxruntime-web` は MIT、互換性問題なし

---

## 推奨アクションプラン

### 即時 (v0.2.3 セキュリティパッチ、半日)

1. **Finding 2 修正**: `loadOpenCv` に URL プロトコル検証追加
2. **Finding 3 修正**: CI workflow に `permissions: {contents: read}` 追加
3. npm publish + GitHub commit

### 後回し可

4. **Finding 1**: vite 6.x への upgrade (devDep のみ、breaking change 確認要)
5. SECURITY.md の作成 (脆弱性報告窓口の明示) — Apache 2.0 プロジェクトとして体裁を整える
6. Dependabot / Renovate の有効化 (依存自動更新 PR)
7. `pnpm audit` を CI に組み込み (regression 防止)
