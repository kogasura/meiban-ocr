// `models/meiban-ocr-vX.onnx` を `src/assets/meiban-ocr-v1.onnx` にコピーする。
// 旧版は base64 で TS 文字列に埋め込んでいたが、4MB の単一リテラルを esbuild が
// 処理できず crash したため、生バイナリを assets として同梱する方式に変更。
//
// バンドル時 default は **新 12-head model (v2-fh, FP16, 582 KB)** を使用。
// 旧 v1 (CRNN+CTC, FP32, 3 MB) も runtime は読めるが、推論時間とサイズで劣るため
// publish には v2-fh を採用する。
//
// asset ファイル名は `meiban-ocr-v1.onnx` のまま (Vite chunk hash 互換性維持、
// runtime 側は output shape で model type を自動判別するためファイル名に依存しない)。
//
// Usage:
//   node scripts/bundle-model.mjs [path/to/source.onnx]
//   # 引数なしで `models/meiban-ocr-v2-fh.onnx` を使用

import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../../..');
// 新 12-head model (推奨) → fallback で旧 CRNN+CTC
const v2Default = resolve(repoRoot, 'models/meiban-ocr-v2-fh.onnx');
const v1Fallback = resolve(repoRoot, 'models/meiban-ocr-v1.onnx');
const defaultSrc = existsSync(v2Default) ? v2Default : v1Fallback;
const src = resolve(process.cwd(), process.argv[2] ?? defaultSrc);
const dst = resolve(__dirname, '../src/assets/meiban-ocr-v1.onnx');

mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
const size = statSync(dst).size;
console.log(`bundle-model: copied ${src} → ${dst} (${(size / 1024).toFixed(1)} KB)`);
