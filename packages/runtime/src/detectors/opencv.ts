/**
 * OpenCV.js を使った古典 text-line 検出器。
 *
 * adaptive threshold → 横方向 morphology closing → contours →
 * アスペクト比とサイズで text-line bbox に絞り込む。
 *
 * Why peer-injected `cv`:
 *   opencv.js は ~10MB あり、本パッケージにバンドルすべきではない。
 *   利用側 (uranus2 etc) が既に opencv.js を持っているケースが多いので、
 *   `createOpenCvDetector(cv)` の形で module を注入する。
 *
 * Expected cv interface (subset of OpenCV.js):
 *   - Mat, matFromImageData, MatVector
 *   - cvtColor, COLOR_RGBA2GRAY
 *   - adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV
 *   - getStructuringElement, Size, MORPH_RECT
 *   - morphologyEx, MORPH_CLOSE
 *   - findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
 *   - boundingRect
 *
 * Usage:
 *   import { createOpenCvDetector } from '@meiban-ocr/runtime/detectors/opencv';
 *   import cv from '@techstark/opencv-js';  // or however the consumer loads OpenCV
 *   const ocr = await MeibanOCR.create({ detector: createOpenCvDetector(cv) });
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import type { BBox, DetectorFn } from './types';

export interface OpenCvDetectorOptions {
  /** 入力最大長辺。これを超えると等比リサイズしてから検出。デフォルト 1280。 */
  maxInputDim?: number;
  /** adaptiveThreshold のブロックサイズ (奇数)。デフォルト 15。 */
  adaptiveBlockSize?: number;
  /** adaptiveThreshold の定数 C。デフォルト 5。 */
  adaptiveC?: number;
  /** 横方向結合カーネル幅。文字を1行 bbox にまとめるのに使う。デフォルト 15。 */
  morphKernelWidth?: number;
  /** 縦方向結合カーネル高さ。デフォルト 3。 */
  morphKernelHeight?: number;
  /** 候補矩形の最小幅 (px)。デフォルト 60。 */
  minWidth?: number;
  /** 候補矩形の最小高さ (px)。デフォルト 10。 */
  minHeight?: number;
  /** 候補矩形のアスペクト比 (width/height) の下限。デフォルト 3 (テキスト行は横長)。 */
  minAspectRatio?: number;
  /** 同上限。デフォルト 20 (長すぎる細線を除外)。 */
  maxAspectRatio?: number;
  /** 候補矩形に縦方向 padding を上下に追加 (px)。CRNN の入力に余白を与える。デフォルト 3。 */
  paddingY?: number;
}

const DEFAULTS: Required<OpenCvDetectorOptions> = {
  maxInputDim: 1280,
  adaptiveBlockSize: 15,
  adaptiveC: 5,
  morphKernelWidth: 15,
  morphKernelHeight: 3,
  minWidth: 60,
  minHeight: 10,
  minAspectRatio: 3,
  maxAspectRatio: 20,
  paddingY: 3,
};

/**
 * OpenCV.js モジュールを受け取り、`DetectorFn` (MeibanOCR に渡せる関数) を生成する。
 */
export function createOpenCvDetector(
  cv: any,
  options: OpenCvDetectorOptions = {},
): DetectorFn {
  const opts: Required<OpenCvDetectorOptions> = { ...DEFAULTS, ...options };

  return function detect(image: ImageData): BBox[] {
    // リサイズ判定
    const scale = computeScale(image.width, image.height, opts.maxInputDim);

    const src = cv.matFromImageData(image);
    let work: any;
    let resized = false;
    if (scale < 1) {
      work = new cv.Mat();
      cv.resize(
        src,
        work,
        new cv.Size(
          Math.round(image.width * scale),
          Math.round(image.height * scale),
        ),
        0,
        0,
        cv.INTER_AREA,
      );
      resized = true;
    } else {
      work = src;
    }

    const gray = new cv.Mat();
    cv.cvtColor(work, gray, cv.COLOR_RGBA2GRAY);

    const binary = new cv.Mat();
    cv.adaptiveThreshold(
      gray,
      binary,
      255,
      cv.ADAPTIVE_THRESH_MEAN_C,
      cv.THRESH_BINARY_INV,
      ensureOdd(opts.adaptiveBlockSize),
      opts.adaptiveC,
    );

    const kernel = cv.getStructuringElement(
      cv.MORPH_RECT,
      new cv.Size(opts.morphKernelWidth, opts.morphKernelHeight),
    );
    const closed = new cv.Mat();
    cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(
      closed,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );

    const bboxes: BBox[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const rect = cv.boundingRect(c);
      c.delete();
      const aspect = rect.width / Math.max(1, rect.height);
      if (
        rect.width >= opts.minWidth &&
        rect.height >= opts.minHeight &&
        aspect >= opts.minAspectRatio &&
        aspect <= opts.maxAspectRatio
      ) {
        // 検出空間 → 画像空間 (resize 戻し) + paddingY
        const inv = scale < 1 ? 1 / scale : 1;
        const x1 = Math.max(0, Math.round(rect.x * inv));
        const y1 = Math.max(0, Math.round((rect.y - opts.paddingY) * inv));
        const x2 = Math.min(
          image.width,
          Math.round((rect.x + rect.width) * inv),
        );
        const y2 = Math.min(
          image.height,
          Math.round((rect.y + rect.height + opts.paddingY) * inv),
        );
        bboxes.push([x1, y1, x2, y2] as const);
      }
    }

    // cleanup (OpenCV.js は Mat を明示 delete する必要)
    src.delete();
    if (resized) work.delete();
    gray.delete();
    binary.delete();
    kernel.delete();
    closed.delete();
    contours.delete();
    hierarchy.delete();

    return bboxes;
  };
}

function computeScale(w: number, h: number, maxDim: number): number {
  const long = Math.max(w, h);
  return long > maxDim ? maxDim / long : 1;
}

function ensureOdd(n: number): number {
  return n % 2 === 0 ? n + 1 : n;
}
