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
  readdirSync,
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
  // 優先順:
  //   1. meiban-ocr-real-v<N>(-suffix)?.onnx の最大 N、 同 N 内では suffix 付きを優先
  //      (例: real-v7-crnn > real-v6-rnn > real-v6 > real-v5)
  //   2. meiban-ocr-v2-fh.onnx (旧 fixed-head)
  //   3. meiban-ocr-v1.onnx (旧 CRNN+CTC)
  //
  // suffix の意図:
  //   - 無印: fixed-head 12-position (旧 default)
  //   - -rnn: fixed-head + BiGRU 増強
  //   - -crnn: 完全な CRNN+CTC (clovaai pretrained 系統)
  //
  // dstName とformat は ONNX の出力 shape (C=13 or 37) で本来判別すべきだが、
  // 配信段で重い ONNX を load せずに manifest を書きたいので filename 規約で代用。
  const modelsDir = resolve(repoRoot, 'models');
  const realCandidates = [];
  if (existsSync(modelsDir)) {
    // -crnn / -rnn / 無印 の suffix を許容
    const realRe = /^meiban-ocr-real-v(\d+)(?:-([a-z]+))?\.onnx$/;
    for (const f of readdirSync(modelsDir)) {
      const m = realRe.exec(f);
      if (m) realCandidates.push({
        path: resolve(modelsDir, f),
        version: parseInt(m[1], 10),
        suffix: m[2] ?? '',
      });
    }
    // version desc、 同 version 内では suffix あり (= 新しい実験変種) を優先
    realCandidates.sort((a, b) => {
      if (a.version !== b.version) return b.version - a.version;
      const aw = a.suffix ? 1 : 0;
      const bw = b.suffix ? 1 : 0;
      return bw - aw;
    });
  }
  const candidates = [
    ...realCandidates.map(c => c.path),
    resolve(repoRoot, 'models/meiban-ocr-v2-fh.onnx'),
    resolve(repoRoot, 'models/meiban-ocr-v1.onnx'),
  ];
  const src = candidates.find(p => existsSync(p));
  if (!src) {
    log('WARN: no custom ONNX model found in models/, skipping custom backend');
    return null;
  }
  // filename 規約で arch を判別: -crnn は CRNN+CTC (C=37)、 それ以外は fixed-head (C=13) 想定
  const srcName = src.split('/').pop();
  const isCrnn = /real-v\d+-crnn\.onnx$/.test(srcName) || srcName === 'meiban-ocr-v1.onnx';
  const dstName = isCrnn ? 'meiban-ocr-crnn.onnx' : 'meiban-ocr-fixed-head.onnx';
  // Why fp32: final(.onnx)は fp16。iOS Safari の WebGPU は shader-f16 未対応のことが多く、
  // さらに onnxruntime-web の WebGPU EP が未対応op(例 LSTM)を wasm に op単位フォールバックする際
  // fp16 は実行時クラッシュする(実機で custom 不発火・paddle fp32 は発火、で確認)。
  // fp32 は WebGPU で paddle 同様に動くため、custom も fp32 を配信する。fp32 兄弟があればそれを使う。
  const fp32Sibling = src.replace(/\.onnx$/, '.fp32.onnx');
  const actualSrc = existsSync(fp32Sibling) ? fp32Sibling : src;
  const dst = resolve(outDir, 'model', 'custom', dstName);
  copyFileSync(actualSrc, dst);
  const size = statSync(dst).size;
  const hash = fileSha256(dst);
  const prec = actualSrc.endsWith('.fp32.onnx') ? 'fp32' : 'fp16';
  log(`copied custom model: ${actualSrc} → ${dst} (${(size / 1024).toFixed(1)} KB, ${prec}, sha256=${hash.slice(0, 16)}…)`);
  return {
    type: 'custom',
    name: dstName,
    source: actualSrc.split('/').pop(),  // 実配信元(fp32)を manifest に記録
    relpath: `model/custom/${dstName}`,
    size,
    hash,
    precision: prec,
    format: isCrnn ? 'crnn-ctc' : 'fixed-head-12pos',
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
  const custom = backends.find(b => b && b.type === 'custom');
  const hasCustom = !!custom;
  const hasPaddle = backends.some(b => b && b.type === 'paddle');
  // custom モデルの実ファイル名/サイズ/format は build 時に決まる(crnn か fixed-head か)。
  // INSTALL の例が実態とズレないよう、 ここから動的に埋める。
  const customUrl = custom
    ? `/assets/meiban-ocr/${custom.relpath}`
    : '/assets/meiban-ocr/model/custom/meiban-ocr-crnn.onnx';
  const customSizeMB = custom ? (custom.size / 1024 / 1024).toFixed(1) : '?';
  const customFmt = custom ? custom.format : 'crnn-ctc';
  const md = `# URANUS2 への統合手順

このディレクトリ (\`dist-uranus2/\`) は **meiban-ocr の URANUS2 統合用ローカル成果物** です。
**npm に publish せず、 GitHub にも commit しない**。 内部運用専用。

## 中身

- \`runtime/\` — Vite build 済の TypeScript bundle (\`index.js\` + \`index.d.ts\`)
- \`model/custom/\` — 自作 OCR モデル (${customSizeMB} MB, ${customFmt})${hasCustom ? '' : ' ← **未同梱** (models/ に custom ONNX が無いため)'}
- \`model/paddle/\` — PaddleOCR PP-OCRv4 mobile (det + rec、 ~15 MB)${hasPaddle ? '' : ' ← **未同梱** (models/ppocrv4_*.onnx が無いため)'}
- \`manifest.json\` — backend ごとのバージョン / ハッシュ / モデルメタ
- \`INSTALL.md\` — このファイル

## URANUS2 側での組み込み

1. このディレクトリ全体を URANUS2 リポジトリの所定の場所にコピー (例: \`assets/meiban-ocr/\`)
2. URANUS2 ビルド時に \`assets/meiban-ocr/runtime/index.js\` を import
3. backend を指定して MeibanOCR.create() を呼ぶ (どちらかを選ぶ or A/B)

### 例: Custom backend(フルフレーム走査 = ハイブリッド: paddle det + custom rec)【推奨】

カメラのフルフレーム(複数銘板)をそのまま渡す運用。paddle det で検出 → custom CRNN で認識する。
(paddle det はシリアル領域を 100% カバー。custom rec は軽量 + Ericsson regex で誤発火を抑える。
paddle rec 単体は辞書6623で重く ~10s かかるため非推奨。)

\`\`\`tsx
import { MeibanOCR, createPaddleDetDetector } from '@assets/meiban-ocr/runtime';

// paddle det を DetectorFn 化(同梱の ppocrv4_det を使う)
// boxMode は default 'quad'(本家準拠の回転矯正crop)。E2E実測(held-out 3,219枚):
//   rect@960 37.7% → quad@960 65.9% → quad@1280 78.4%(detLongSide はレイテンシと相談)
const detector = await createPaddleDetDetector({
  detModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_det.onnx',
  executionProviders: ['webgpu', 'wasm'],
  detLongSide: 1280,   // 精度優先。フレームレート優先なら 960
});

const ocr = await MeibanOCR.create({
  backend: 'custom',
  modelUrl: '${customUrl}',
  vendor: 'ericsson',
  detector,            // ★ paddle det 検出 → custom CRNN 認識(ハイブリッド)
  prefilter: false,    // paddle det の box をそのまま使う
  recenter: false,
  minConfidence: 0.5,
  executionProviders: ['webgpu', 'wasm'],   // custom モデルは fp32(WebGPU/WASM 両対応)
});

const results = await ocr.recognize(cameraFullFrame);
\`\`\`

> reticle(ユーザーが1枚を枠に収める)UX の場合のみ、 detector を full-frame
> \`(img) => [[0, 0, img.width, img.height]]\` にし、 アプリ側で reticle 枠内だけを crop して渡す。
> ※ カメラのフルフレームを full-frame detector に渡すと全景が潰れて不発火するので注意。
\`\`\`

### 例: Paddle backend (PP-OCRv4、 訓練不要、 大きめ)

\`\`\`tsx
const ocr = await MeibanOCR.create({
  backend: 'paddle',
  detModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_det.onnx',
  recModelUrl: '/assets/meiban-ocr/model/paddle/ppocrv4_rec.onnx',
  executionProviders: ['webgpu', 'wasm'],
  minConfidence: 0.5,
  // 汎用 det が背景の文字様パターンを大量検出すると B(rec バッチ)が膨らみ、 メインスレッド
  // (rec 推論 + box毎前処理 + 大きな CTC デコード)を占有して UI がフリーズする。 銘板スキャナは
  // 主要テキスト領域だけ読めれば十分なので、 rec に渡す box を面積上位 K 件に制限する。
  maxRecBoxes: 4,
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
