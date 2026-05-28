# `@meiban-ocr/runtime` v0.1.0 npm publish 手順

最終確認 + 公開コマンド。ユーザー側で実行する。

---

## 事前確認 (1 分)

```bash
cd /home/yuuki-okubo/meiban-ocr/packages/runtime

# 既ビルド + tarball が最新か
pnpm build                                    # → dist/ 再生成
pnpm pack                                     # → meiban-ocr-runtime-0.1.0.tgz

# tarball 内容を最終チェック
tar -tzf meiban-ocr-runtime-0.1.0.tgz | sort
# 期待: package/{LICENSE, README.md, package.json, dist/*.{js,d.ts}}

# ローカル smoke (オプション)
mkdir -p /tmp/meiban-smoke && cd /tmp/meiban-smoke
echo '{"type":"module"}' > package.json
npm install ~/meiban-ocr/packages/runtime/meiban-ocr-runtime-0.1.0.tgz onnxruntime-web
node -e "import('@meiban-ocr/runtime').then(m => console.log(typeof m.MeibanOCR.create))"
# → "function" が出れば OK
```

---

## npm 認証

scope `@meiban-ocr` を **取得していない場合は、まず npm Web UI で organization 作成**:

1. <https://www.npmjs.com/login> でログイン
2. <https://www.npmjs.com/org/create> で organization `meiban-ocr` を作成
   - free plan で OK (公開パッケージのみ可)
3. ローカル CLI で login:

```bash
npm login
# username, password, email を対話入力
# 2FA があれば OTP も
```

確認:

```bash
npm whoami
# → 自分の username が表示される
```

---

## publish 実行

```bash
cd /home/yuuki-okubo/meiban-ocr/packages/runtime

# dry-run (実際には publish しない、何が送られるか見る)
npm publish --dry-run

# 本番 publish
npm publish --access public
```

`publishConfig.access: public` を package.json に設定済みなので `--access public` は
保険。scoped package は default で private 扱いだが、これで public 公開される。

成功時の出力例:

```
npm notice
npm notice 📦  @meiban-ocr/runtime@0.1.0
npm notice === Tarball Contents ===
npm notice ... (ファイル一覧) ...
npm notice === Tarball Details ===
npm notice name:          @meiban-ocr/runtime
npm notice version:       0.1.0
npm notice ...
+ @meiban-ocr/runtime@0.1.0
```

---

## publish 後

```bash
# レジストリで確認
npm view @meiban-ocr/runtime

# 一度別 dir でインストールテスト
mkdir -p /tmp/meiban-published-check && cd $_
npm install @meiban-ocr/runtime onnxruntime-web
node -e "import('@meiban-ocr/runtime').then(m => console.log(m.ericsson.strictRegex))"
```

---

## バージョン更新 (今後)

```bash
cd /home/yuuki-okubo/meiban-ocr/packages/runtime
npm version patch   # 0.1.0 → 0.1.1 (バグ修正)
npm version minor   # 0.1.0 → 0.2.0 (機能追加)
npm version major   # 0.1.0 → 1.0.0 (破壊的変更)
pnpm build
npm publish --access public
```

`npm version` は package.json を書き換え + git tag を作る。
git に commit しないなら `--no-git-tag-version` を付ける。

---

## 公開取り下げ (緊急時のみ)

```bash
# 72時間以内かつ、誰もインストールしていなければ可
npm unpublish @meiban-ocr/runtime@0.1.0
```

72 時間超えた場合は deprecate のみ:

```bash
npm deprecate @meiban-ocr/runtime@0.1.0 "Use 0.1.1+ for fix"
```

---

## チェックリスト

- [ ] `pnpm build` 成功 (dist/ 生成)
- [ ] `pnpm pack` で tarball 3.0 MB 程度
- [ ] tarball 内に LICENSE / README.md / dist/index.js / dist/meiban-ocr-v1-*.js が入っている
- [ ] スコープ `@meiban-ocr` を npm Web UI で取得済み
- [ ] `npm whoami` で自分が表示される
- [ ] `npm publish --dry-run` でエラー無し
- [ ] `npm publish --access public` 実行
- [ ] `npm view @meiban-ocr/runtime` で 0.1.0 が見える
- [ ] 別ディレクトリで `npm install @meiban-ocr/runtime` が成功する
