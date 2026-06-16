// URANUS2 統合向けの local 配信成果物を `dist-uranus2/` に出力する。
//
// 出力物:
//   - dist-uranus2/runtime/         ← Vite build した JS bundle + .d.ts
//   - dist-uranus2/model/paddle/    ← 同梱の PaddleOCR ONNX (det + rec)
//   - dist-uranus2/INSTALL.md       ← URANUS2 側での統合手順
//   - dist-uranus2/manifest.json    ← バージョン / モデルメタ / ハッシュ
//
// この成果物は **絶対に commit / npm publish しない**。
//   - dist-uranus2/ は .gitignore で除外済
//   - packages/runtime/package.json は private:true で npm publish 物理ブロック
//
// URANUS2 への配信は **手動コピー** で運用:
//   1. このスクリプトで dist-uranus2/ を生成
//   2. tar / zip にまとめて URANUS2 リポジトリ or 社内 CDN に upload
//   3. URANUS2 側で det/rec ModelUrl を指す or assets として参照
//
// 2026-06-16 custom backend (自作 12-head/CRNN) 廃止 (vendor-setting-client#430)。
// backend は paddle (PP-OCRv4 det + rec) 単独。
//
// Usage:
//   pnpm build:uranus2
//   # or
//   node scripts/build-for-uranus2.mjs

import { execSync } from 'node:child_process';
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createHash as cryptoCreateHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const runtimeDir = resolve(__dirname, '..');
const repoRoot = resolve(runtimeDir, '../..');
const outDir = resolve(repoRoot, 'dist-uranus2');

function log(msg) {
  process.stdout.write(`[build-for-uranus2] ${msg}\n`);
}

function fileSha256(path) {
  const h = cryptoCreateHash('sha256');
  h.update(readFileSync(path));
  return h.digest('hex');
}

function clean() {
  if (existsSync(outDir)) {
    rmSync(outDir, { recursive: true, force: true });
  }
  mkdirSync(outDir, { recursive: true });
  mkdirSync(resolve(outDir, 'runtime'), { recursive: true });
  mkdirSync(resolve(outDir, 'model', 'paddle'), { recursive: true });
}

function buildRuntime() {
  log('vite build + tsc (declaration)');
  execSync('pnpm run build', { cwd: runtimeDir, stdio: 'inherit' });
}

function copyRuntime() {
  const distDir = resolve(runtimeDir, 'dist');
  if (!existsSync(distDir)) {
    throw new Error(`runtime dist/ not found at ${distDir} — build must have failed`);
  }
  cpSync(distDir, resolve(outDir, 'runtime'), { recursive: true });
  log(`copied runtime/ ← ${distDir}`);
}

function copyPaddleModels() {
  const detSrc = resolve(repoRoot, 'models/ppocrv4_det.onnx');
  const recSrc = resolve(repoRoot, 'models/ppocrv4_rec.onnx');
  if (!existsSync(detSrc) || !existsSync(recSrc)) {
    log(`WARN: paddle models not found in models/ (expected ppocrv4_det.onnx + ppocrv4_rec.onnx), skipping paddle backend`);
    return null;
  }
  const detDst = resolve(outDir, 'model', 'paddle', 'ppocrv4_det.onnx');
  const recDst = resolve(outDir, 'model', 'paddle', 'ppocrv4_rec.onnx');
  copyFileSync(detSrc, detDst);
  copyFileSync(recSrc, recDst);
  const detSize = statSync(detDst).size;
  const recSize = statSync(recDst).size;
  const detHash = fileSha256(detDst);
  const recHash = fileSha256(recDst);
  log(`copied paddle det: ${detSrc} → ${detDst} (${(detSize / 1024).toFixed(1)} KB, sha256=${detHash.slice(0, 16)}…)`);
  log(`copied paddle rec: ${recSrc} → ${recDst} (${(recSize / 1024).toFixed(1)} KB, sha256=${recHash.slice(0, 16)}…)`);
  return {
    type: 'paddle',
    files: {
      det: { relpath: 'model/paddle/ppocrv4_det.onnx', size: detSize, hash: detHash },
      rec: { relpath: 'model/paddle/ppocrv4_rec.onnx', size: recSize, hash: recHash },
    },
    format: 'paddleocr-ppocrv4-mobile',
    size_bytes_total: detSize + recSize,
  };
}

function writeManifest(backends) {
  const pkg = JSON.parse(readFileSync(resolve(runtimeDir, 'package.json'), 'utf-8'));
  const manifest = {
    name: pkg.name,
    version: pkg.version,
    distribution: 'internal-uranus2',
    built_at: new Date().toISOString(),
    runtime: {
      entry: 'runtime/index.js',
      types: 'runtime/index.d.ts',
    },
    backends: backends.filter(Boolean),
    notes: [
      'This artifact is NOT for npm publication or public distribution.',
      'Distribute only to URANUS2 internal deployment.',
      'Backend is PaddleOCR PP-OCRv4 (det + rec). custom backend was removed in vendor-setting-client#430.',
    ],
  };
  const path = resolve(outDir, 'manifest.json');
  writeFileSync(path, JSON.stringify(manifest, null, 2) + '\n');
  log(`wrote manifest.json (backends: ${backends.filter(Boolean).map(b => b.type).join(', ') || 'none'})`);
}

function writeInstallGuide(backends) {
  const hasPaddle = backends.some(b => b && b.type === 'paddle');
  const md = `# URANUS2 への統合手順

このディレクトリ (\`dist-uranus2/\`) は **meiban-ocr の URANUS2 統合用ローカル成果物** です。
**npm に publish せず、 GitHub にも commit しない**。 内部運用専用。

## 中身

- \`runtime/\` — Vite build 済の TypeScript bundle (\`index.js\` + \`index.d.ts\`)
- \`model/paddle/\` — PaddleOCR PP-OCRv4 mobile (det + rec、 ~15 MB)${hasPaddle ? '' : ' ← **未同梱** (models/ppocrv4_*.onnx が無いため)'}
- \`manifest.json\` — backend のバージョン / ハッシュ / モデルメタ
- \`INSTALL.md\` — このファイル

## URANUS2 側での組み込み

1. このディレクトリ全体を URANUS2 リポジトリの所定の場所にコピー (例: \`assets/meiban-ocr/\`)
2. URANUS2 ビルド時に \`assets/meiban-ocr/runtime/index.js\` を import
3. det + rec モデルを指定して MeibanOCR.create() を呼ぶ

### 例: Paddle backend (PP-OCRv4、 訓練不要)

\`\`\`tsx
import { MeibanOCR } from '@assets/meiban-ocr/runtime';

const ocr = await MeibanOCR.create({
  backend: 'paddle',   // default かつ現状唯一
  detModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_det.onnx',
  recModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_rec.onnx',
  vendor: 'ericsson',
  executionProviders: ['webgpu', 'wasm'],
  minConfidence: 0.5,
  // 汎用 det が背景の文字様パターンを大量検出すると B(rec バッチ)が膨らみ、 メインスレッド
  // (rec 推論 + box毎前処理 + 大きな CTC デコード)を占有して UI がフリーズする。 銘板スキャナは
  // 主要テキスト領域だけ読めれば十分なので、 rec に渡す box を面積上位 K 件に制限する。
  maxRecBoxes: 4,
  // det の長辺リサイズ。 精度優先なら 1280、 フレームレート優先なら 960 (default)。
  detLongSide: 960,
});

const results = await ocr.recognize(cameraFrame);
\`\`\`

### モバイル(メモリ制約)プロファイル

iOS Safari / WKWebView では ORT セッションの wasm ヒープが大きい (実測 settle RSS:
det@1280=+268MB / @960=+219MB / @640=+171MB)。 WebGPU 併用はバッファ二重持ちの恐れがあるため
wasm 固定 + 長辺を絞るのを推奨:

\`\`\`tsx
const ocr = await MeibanOCR.create({
  backend: 'paddle',
  detModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_det.onnx',
  recModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_rec.onnx',
  vendor: 'ericsson',
  executionProviders: ['wasm'],   // WebGPU 併用はバッファ二重持ちの恐れ。 wasm 固定
  detLongSide: 640,               // reticle UX (銘板が画面大) なら 640 で十分
  maxRecBoxes: 4,
  minConfidence: 0.5,
});
\`\`\`

## 配信時の注意

- このディレクトリは git で commit してはいけない。 ルート \`.gitignore\` の \`/dist-uranus2/\` で除外済。
- URANUS2 リポジトリへの配置は **手動コピー or 内部 CDN 経由**。 公開 CDN は不可。
- Paddle backend のモデルは公開済 OSS (PaddleOCR Apache-2.0)、 訓練データ漏洩リスクなし。
`;
  writeFileSync(resolve(outDir, 'INSTALL.md'), md);
  log('wrote INSTALL.md');
}

function summarize(backends) {
  log('');
  log('========== build complete ==========');
  log(`output: ${outDir}`);
  for (const b of backends.filter(Boolean)) {
    if (b.type === 'paddle') {
      log(`  paddle: det+rec (${((b.size_bytes_total) / 1024).toFixed(1)} KB total)`);
    }
  }
  const present = backends.filter(Boolean).map(b => b.type);
  if (present.length === 0) {
    log('  WARN: no backend models found! Upload was empty.');
  }
  log('');
  log('Next: upload dist-uranus2/ to URANUS2 (manual copy or internal CDN).');
  log('Do NOT commit or npm publish.');
}

function main() {
  log(`runtime: ${runtimeDir}`);
  log(`output:  ${outDir}`);
  clean();
  buildRuntime();
  copyRuntime();
  const backends = [copyPaddleModels()];
  writeManifest(backends);
  writeInstallGuide(backends);
  summarize(backends);
}

try {
  main();
} catch (err) {
  process.stderr.write(`[build-for-uranus2] ERROR: ${err.message}\n`);
  process.exit(1);
}
