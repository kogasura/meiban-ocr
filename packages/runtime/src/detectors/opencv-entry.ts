/**
 * sub-export entry: `@meiban-ocr/runtime/detectors/opencv`
 */

export {
  createOpenCvDetector,
  loadOpenCv,
  type LoadOpenCvOptions,
  type OpenCvDetectorOptions,
} from './opencv';
export type { BBox, DetectorFn } from './types';
