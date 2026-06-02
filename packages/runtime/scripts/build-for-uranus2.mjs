// URANUS2 統合向けの local 配信成果物を `dist-uranus2/` に出力する。
//
// 出力物:
//   - dist-uranus2/runtime/         ← Vite build した JS bundle + .d.ts
//   - dist-uranus2/model/           ← 同梱の ONNX (FP16 / FP32 から FP16 を採用)
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
//   3. URANUS2 側で modelUrl を指す or assets として参照
//
// Usage:
//   pnpm build:uranus2
//   # or
//   node scripts/build-for-uranus2.mjs

import { execSync } from 'node:child_process';
import {
  copyFileSync,
  cpSync,
  createHash,
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
  mkdirSync(resolve(outDir, 'model'), { recursive: true });
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

function copyModel() {
  // 優先: 12-head v2-fh、 次善: 旧 v1 CRNN
  const candidates = [
    resolve(repoRoot, 'models/meiban-ocr-v2-fh.onnx'),
    resolve(repoRoot, 'models/meiban-ocr-v1.onnx'),
  ];
  const src = candidates.find(p => existsSync(p));
  if (!src) {
    throw new Error(
      `no ONNX model found in models/. Expected one of:\n  ${candidates.join('\n  ')}`,
    );
  }
  const dstName = src.endsWith('v2-fh.onnx') ? 'meiban-ocr-fixed-head.onnx' : 'meiban-ocr-crnn.onnx';
  const dst = resolve(outDir, 'model', dstName);
  copyFileSync(src, dst);
  const size = statSync(dst).size;
  const hash = fileSha256(dst);
  log(`copied model: ${src} → ${dst} (${(size / 1024).toFixed(1)} KB, sha256=${hash.slice(0, 16)}…)`);
  return { src, dst, size, hash, name: dstName };
}

function writeManifest(modelInfo) {
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
    model: {
      file: `model/${modelInfo.name}`,
      sha256: modelInfo.hash,
      size_bytes: modelInfo.size,
      format: modelInfo.name.includes('fixed-head') ? 'fixed-head-12pos' : 'crnn-ctc',
    },
    notes: [
      'This artifact is NOT for npm publication or public distribution.',
      'It may contain a model trained on real customer serial numbers.',
      'Distribute only to URANUS2 internal deployment.',
    ],
  };
  const path = resolve(outDir, 'manifest.json');
  writeFileSync(path, JSON.stringify(manifest, null, 2) + '\n');
  log(`wrote manifest.json (model=${modelInfo.name})`);
}

function writeInstallGuide() {
  const md = `# URANUS2 への統合手順

このディレクトリ (\`dist-uranus2/\`) は **meiban-ocr の URANUS2 統合用ローカル成果物** です。
**npm に publish せず、 GitHub にも commit しない**。 内部運用専用。

## 中身

- \`runtime/\` — Vite build 済の TypeScript bundle (\`index.js\` + \`index.d.ts\`)
- \`model/\` — 同梱の ONNX (FP16 量子化済、 fixed-head 12-position)
- \`manifest.json\` — version / ハッシュ / モデルメタ
- \`INSTALL.md\` — このファイル

## URANUS2 側での組み込み (推奨フロー)

1. このディレクトリを URANUS2 リポジトリの所定の場所にコピー (例: \`assets/meiban-ocr/\`)
2. URANUS2 ビルド時に \`assets/meiban-ocr/runtime/index.js\` を import
3. \`MeibanOCR.create({ modelUrl: '/assets/meiban-ocr/model/meiban-ocr-fixed-head.onnx' })\` で初期化

### 例: React コンポーネント

\`\`\`tsx
import { MeibanOCR } from '@assets/meiban-ocr/runtime';

const ocr = await MeibanOCR.create({
  modelUrl: '/assets/meiban-ocr/model/meiban-ocr-fixed-head.onnx',
  executionProviders: ['webgpu', 'wasm'],
  minConfidence: 0.7,
});

const results = await ocr.recognize(videoFrame);
\`\`\`

## 配信時の注意

- このディレクトリは git で commit してはいけない。 ルート \`.gitignore\` の \`/dist-uranus2/\` で除外済。
- URANUS2 リポジトリへの配置は **手動コピー or 内部 CDN 経由**。 公開 CDN は不可。
- モデルファイル (\`model/*.onnx\`) は実シリアルを認識する内部モデルを含む可能性。
  顧客向け配信時はアクセス制限を確認。
`;
  writeFileSync(resolve(outDir, 'INSTALL.md'), md);
  log('wrote INSTALL.md');
}

function summarize(modelInfo) {
  log('');
  log('========== build complete ==========');
  log(`output: ${outDir}`);
  log(`model:  ${modelInfo.name} (${(modelInfo.size / 1024).toFixed(1)} KB)`);
  log(`hash:   ${modelInfo.hash}`);
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
  const modelInfo = copyModel();
  writeManifest(modelInfo);
  writeInstallGuide();
  summarize(modelInfo);
}

try {
  main();
} catch (err) {
  process.stderr.write(`[build-for-uranus2] ERROR: ${err.message}\n`);
  process.exit(1);
}
