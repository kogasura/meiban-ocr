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
export type { ImageInput } from './preprocess';
export {
  detBoxBBox,
  detBoxQuad,
  type BBox,
  type DetBox,
  type Quad,
  type QuadBox,
} from './detectors/types';
