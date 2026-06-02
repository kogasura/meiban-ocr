/**
 * Backend factory — backend type を見て該当実装を返す。
 *
 * 使い方:
 *   const backend = await createBackend('custom', { modelUrl, ... });
 *   const backend = await createBackend('paddle', { detModelUrl, recModelUrl, ... });
 */

import { CustomBackend } from './custom';
import type {
  AnyBackendInit,
  Backend,
  BackendType,
  CustomBackendInit,
  PaddleBackendInit,
} from './types';

export async function createBackend(
  type: BackendType,
  options: AnyBackendInit = {},
): Promise<Backend> {
  switch (type) {
    case 'custom':
      return CustomBackend.create(options as CustomBackendInit);
    case 'paddle': {
      // 動的 import で paddle backend を必要時のみロード
      // (custom-only 利用時は paddle のコードを bundle から除外可能、 将来の最適化)
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
