/**
 * Backend 実装で共有する小物 (URL 検証など)。
 * 各 backend がインポートする内部ユーティリティ。 public export しない。
 */

// Why: ORT は `data:` / `blob:` / `https:` / `http:` などを受け付ける。
// 利用側が untrusted な値 (URL query 等) を modelUrl に渡したとき、
// `javascript:` / `vbscript:` / `file:` が来ると任意 JS 実行 や local file 読込
// につながる可能性があるため、whitelist で検証する。
const ALLOWED_MODEL_URL_PROTOCOLS = new Set([
  'https:',
  'http:',
  'data:',
  'blob:',
]);

export function validateModelUrl(rawUrl: string, fieldName: string = 'modelUrl'): void {
  let parsed: URL;
  try {
    const base =
      typeof location !== 'undefined' && location.href
        ? location.href
        : 'http://localhost/';
    parsed = new URL(rawUrl, base);
  } catch {
    throw new Error(`MeibanOCR.create: invalid ${fieldName}: ${rawUrl}`);
  }
  if (!ALLOWED_MODEL_URL_PROTOCOLS.has(parsed.protocol)) {
    throw new Error(
      `MeibanOCR.create: unsupported protocol "${parsed.protocol}" in ${fieldName}. ` +
        `Allowed: http, https, data, blob.`,
    );
  }
}

import * as ort from 'onnxruntime-web';

/**
 * ORT session を作る共通ヘルパ。 bytes 優先、 次に URL、 最後に default URL。
 */
export async function createOrtSession(
  modelBytes: Uint8Array | ArrayBuffer | undefined,
  modelUrl: string | undefined,
  defaultUrl: string | undefined,
  sessionOptions: ort.InferenceSession.SessionOptions,
  fieldName: string = 'modelUrl',
): Promise<ort.InferenceSession> {
  if (modelBytes) {
    const bytes =
      modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
    return ort.InferenceSession.create(bytes, sessionOptions);
  }
  const url = modelUrl ?? defaultUrl;
  if (!url) {
    throw new Error(`MeibanOCR.create: ${fieldName} or bytes required`);
  }
  if (url !== defaultUrl) {
    validateModelUrl(url, fieldName);
  }
  return ort.InferenceSession.create(url, sessionOptions);
}
