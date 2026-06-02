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
  // 2 系統の model 配置: custom (自作 12-head) と paddle (PP-OCRv4)
  mkdirSync(resolve(outDir, 'model', 'custom'), { recursive: true });
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

function copyCustomModel() {
  // 優先: 12-head v2-fh、 次善: 旧 v1 CRNN
  const candidates = [
    resolve(repoRoot, 'models/meiban-ocr-v2-fh.onnx'),
    resolve(repoRoot, 'models/meiban-ocr-v1.onnx'),
  ];
  const src = candidates.find(p => existsSync(p));
  if (!src) {
    log('WARN: no custom ONNX model found in models/, skipping custom backend');
    return null;
  }
  const dstName = src.endsWith('v2-fh.onnx')
    ? 'meiban-ocr-fixed-head.onnx'
    : 'meiban-ocr-crnn.onnx';
  const dst = resolve(outDir, 'model', 'custom', dstName);
  copyFileSync(src, dst);
  const size = statSync(dst).size;
  const hash = fileSha256(dst);
  log(`copied custom model: ${src} → ${dst} (${(size / 1024).toFixed(1)} KB, sha256=${hash.slice(0, 16)}…)`);
  return {
    type: 'custom',
    name: dstName,
    relpath: `model/custom/${dstName}`,
    size,
    hash,
    format: src.endsWith('v2-fh.onnx') ? 'fixed-head-12pos' : 'crnn-ctc',
  };
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
      'It may contain a model trained on real customer serial numbers.',
      'Distribute only to URANUS2 internal deployment.',
      'Backends are A/B testable via MeibanOCR.create({ backend: "custom" | "paddle" }).',
    ],
  };
  const path = resolve(outDir, 'manifest.json');
  writeFileSync(path, JSON.stringify(manifest, null, 2) + '\n');
  log(`wrote manifest.json (backends: ${backends.filter(Boolean).map(b => b.type).join(', ')})`);
}

function writeInstallGuide(backends) {
  const hasCustom = backends.some(b => b && b.type === 'custom');
  const hasPaddle = backends.some(b => b && b.type === 'paddle');
  const md = `# URANUS2 への統合手順

このディレクトリ (\`dist-uranus2/\`) は **meiban-ocr の URANUS2 統合用ローカル成果物** です。
**npm に publish せず、 GitHub にも commit しない**。 内部運用専用。

## 中身

- \`runtime/\` — Vite build 済の TypeScript bundle (\`index.js\` + \`index.d.ts\`)
- \`model/custom/\` — 自作 12-head OCR モデル (~580 KB)${hasCustom ? '' : ' ← **未同梱** (models/meiban-ocr-v2-fh.onnx が無いため)'}
- \`model/paddle/\` — PaddleOCR PP-OCRv4 mobile (det + rec、 ~15 MB)${hasPaddle ? '' : ' ← **未同梱** (models/ppocrv4_*.onnx が無いため)'}
- \`manifest.json\` — backend ごとのバージョン / ハッシュ / モデルメタ
- \`INSTALL.md\` — このファイル

## URANUS2 側での組み込み

1. このディレクトリ全体を URANUS2 リポジトリの所定の場所にコピー (例: \`assets/meiban-ocr/\`)
2. URANUS2 ビルド時に \`assets/meiban-ocr/runtime/index.js\` を import
3. backend を指定して MeibanOCR.create() を呼ぶ (どちらかを選ぶ or A/B)

### 例: Custom backend (自作 12-head、 軽量、 訓練要)

\`\`\`tsx
import { MeibanOCR } from '@assets/meiban-ocr/runtime';

const ocr = await MeibanOCR.create({
  backend: 'custom',
  modelUrl: '/assets/meiban-ocr/model/custom/meiban-ocr-fixed-head.onnx',
  executionProviders: ['webgpu', 'wasm'],
  minConfidence: 0.7,
});

const results = await ocr.recognize(videoFrame);
\`\`\`

### 例: Paddle backend (PP-OCRv4、 訓練不要、 大きめ)

\`\`\`tsx
const ocr = await MeibanOCR.create({
  backend: 'paddle',
  detModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_det.onnx',
  recModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_rec.onnx',
  executionProviders: ['webgpu', 'wasm'],
  minConfidence: 0.5,
});

const results = await ocr.recognize(videoFrame);
\`\`\`

### 例: A/B 比較

\`\`\`tsx
// 設定で切替
const backend = config.OCR_BACKEND;  // 'custom' | 'paddle'
const ocr = await MeibanOCR.create(
  backend === 'paddle'
    ? { backend: 'paddle', detModelUrl: '...', recModelUrl: '...' }
    : { backend: 'custom', modelUrl: '...' }
);

// もしくは両方並べて同じ frame に流して結果比較
const [a, b] = await Promise.all([customOcr.recognize(frame), paddleOcr.recognize(frame)]);
\`\`\`

## 配信時の注意

- このディレクトリは git で commit してはいけない。 ルート \`.gitignore\` の \`/dist-uranus2/\` で除外済。
- URANUS2 リポジトリへの配置は **手動コピー or 内部 CDN 経由**。 公開 CDN は不可。
- Custom backend のモデルは **実シリアルを認識する可能性** がある (訓練データ次第)。
  顧客向け配信時はアクセス制限を確認。
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
    if (b.type === 'custom') {
      log(`  custom: ${b.name} (${(b.size / 1024).toFixed(1)} KB)`);
    } else if (b.type === 'paddle') {
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
  const backends = [copyCustomModel(), copyPaddleModels()];
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
