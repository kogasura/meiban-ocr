/**
 * 後方互換 re-export: 旧 `src/detector.ts` の API を `src/detectors/` に移した。
 * 既存利用側は `import { collectWindowBoxes, nmsByText } from '@meiban-ocr/runtime'`
 * のままで動作する。
 */

export {
  collectWindowBoxes,
  computeDownscale,
  createSlidingWindowDetector,
  generateWindowBoxes,
  type ImageSize,
  type SlidingWindowOptions,
} from './detectors/sliding-window';

export { iou, nmsByText, type ScoredDetection } from './detectors/nms';

// 旧名互換: ProposalGeneratorOptions → SlidingWindowOptions
export type { SlidingWindowOptions as ProposalGeneratorOptions } from './detectors/sliding-window';
