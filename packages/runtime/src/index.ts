export {
  MeibanOCR,
  type MeibanOCROptions,
  type OCRResult,
} from './MeibanOCR';
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
export type { BBox, DetectorFn } from './detectors/types';
export {
  createSlidingWindowDetector,
  type SlidingWindowOptions,
} from './detectors/sliding-window';
