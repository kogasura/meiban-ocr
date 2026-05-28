// `models/meiban-ocr-v1.onnx` を `src/assets/` にコピーする。
// 旧版は base64 で TS 文字列に埋め込んでいたが、4MB の単一リテラルを esbuild が
// 処理できず crash したため、生バイナリを assets として同梱する方式に変更。
//
// Usage:
//   node scripts/bundle-model.mjs [path/to/source.onnx]

import { copyFileSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../../..');
const defaultSrc = resolve(repoRoot, 'models/meiban-ocr-v1.onnx');
const src = resolve(process.cwd(), process.argv[2] ?? defaultSrc);
const dst = resolve(__dirname, '../src/assets/meiban-ocr-v1.onnx');

mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
const size = statSync(dst).size;
console.log(`bundle-model: copied ${src} → ${dst} (${(size / 1024).toFixed(1)} KB)`);
