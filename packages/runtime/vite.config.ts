import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  // Why: .onnx は Vite が知らない拡張子なので明示的に asset として認識させる。
  // これにより `import url from './model.onnx?url'` がビルド時に解決され、
  // ファイルは `dist/assets/meiban-ocr-v1-<hash>.onnx` のように出力される。
  assetsInclude: ['**/*.onnx'],
  build: {
    lib: {
      // multi-entry: main + opencv detector sub-export
      // 利用側は `@meiban-ocr/runtime` か `@meiban-ocr/runtime/detectors/opencv` で
      // それぞれ参照、ツリーシェイクが効く構造。
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        'detectors/opencv-entry': resolve(
          __dirname,
          'src/detectors/opencv-entry.ts',
        ),
      },
      // Why ESM only: CJS で出すと 4MB のモデルチャンクが二重に生成され
      // tarball が肥大化する。Node 20+/モダンbundlerは ESM対応。
      formats: ['es'],
    },
    rollupOptions: {
      external: ['onnxruntime-web'],
      output: {
        globals: { 'onnxruntime-web': 'ort' },
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
    // Why no sourcemap: 8MB を超え tarball を肥大化させる。
    sourcemap: false,
    target: 'es2022',
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
