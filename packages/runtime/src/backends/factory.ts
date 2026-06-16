/**
 * Backend factory — backend type を見て該当実装を返す。
 *
 * 使い方:
 *   const backend = await createBackend('paddle', { detModelUrl, recModelUrl, ... });
 *
 * 現状 backend は paddle (PP-OCRv4 det + rec) のみ。custom (自作 12-head/CRNN) は
 * vendor-setting-client#430 で廃止。将来 backend を追加する際は case を足す。
 */

import type {
  AnyBackendInit,
  Backend,
  BackendType,
  PaddleBackendInit,
} from './types';

export async function createBackend(
  type: BackendType,
  options: AnyBackendInit = {},
): Promise<Backend> {
  switch (type) {
    case 'paddle': {
      // 動的 import で paddle backend を必要時のみロード。
      const { PaddleBackend } = await import('./paddle');
      return PaddleBackend.create(options as PaddleBackendInit);
    }
    default: {
      // exhaustiveness check
      const _exhaustive: never = type;
      throw new Error(`unknown backend type: ${String(_exhaustive)}`);
    }
  }
}
