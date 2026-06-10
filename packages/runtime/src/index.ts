export {
  MeibanOCR,
  type MeibanOCROptions,
  type OCRResult,
} from './MeibanOCR';
export type {
  AnyBackendInit,
  Backend,
  BackendType,
  CommonBackendOptions,
  CustomBackendInit,
  PaddleBackendInit,
} from './backends/types';
export {
  BLANK_IDX,
  CHARSET,
  INPUT_HEIGHT,
  INPUT_WIDTH,
  NUM_CLASSES,
} from './constants';
export { ericsson, type VendorPattern, VENDOR_PATTERNS } from './vendors';
export { ctcGreedyDecode, applyCorrectionPipeline, preprocessText } from './decoder';
export type { CropOptions, ImageInput, RecenterOptions } from './preprocess';
export {
  detBoxBBox,
  detBoxQuad,
  type BBox,
  type DetBox,
  type DetectorFn,
  type Quad,
  type QuadBox,
} from './detectors/types';
export {
  createSlidingWindowDetector,
  type SlidingWindowOptions,
} from './detectors/sliding-window';
export {
  createPaddleDetDetector,
  type PaddleDetDetectorOptions,
} from './detectors/paddle-det';
