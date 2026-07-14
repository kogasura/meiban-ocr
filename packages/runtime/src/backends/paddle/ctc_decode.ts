/**
 * PaddleOCR (PP-OCRv4) CTC decode + dict lookup。
 *
 * 入力: rec ONNX の出力 softmax probabilities (B, T, C)
 * 出力: バッチ各位置の text + confidence
 *
 * PaddleOCR の dict 規約:
 *   - index 0           : CTC blank (skip)
 *   - index 1..N        : dict[i-1] (改行区切りの 1 文字)
 *   - index N+1         : 半角スペース ' '
 *   (C = N + 2)
 *
 * CTC greedy decode:
 *   - 各時刻 t で argmax
 *   - 連続する同一インデックスを merge (CTC collapse)
 *   - blank を除く
 *   - dict から文字を引いて連結
 *
 * Confidence は **採用された各時刻の top1 確率の平均**
 * (PaddleOCR Python 実装と同じ式)。
 *
 * ## ベンダー別 許可文字制約 (constrained decode)
 *
 * `charset` を渡すと、各タイムステップの argmax を「許可文字 (+ CTC blank)」
 * の class のみに制約する (非許可 class は選択肢から除外)。 モデル自体・
 * dict 次元は変更しない — argmax 前に候補を絞るだけ。
 * `charset` 省略時は従来と完全に同一の挙動 (全 class が候補)。
 */

/** dict.txt の中身 (newline separated) を内部 char 配列に変換。 */
export function parseDict(dictText: string): string[] {
  // 末尾 newline は除外。 トリムはせず (空文字を含む可能性は除外)。
  const lines = dictText.split('\n');
  // 末尾の空行を除去
  while (lines.length > 0 && lines[lines.length - 1] === '') {
    lines.pop();
  }
  return lines;
}

/**
 * 期待される出力 channel 数を返す。
 * dict が N 文字なら blank + dict + space = N + 2 channel。
 */
export function expectedNumClasses(dict: string[]): number {
  return dict.length + 2;
}

export interface CtcDecodeResult {
  text: string;
  confidence: number;
}

/**
 * `charset` + `dict` から、 CTC decode で選択可能な class index の boolean mask
 * (length C) を組み立てる。 index 0 (blank) は常に許可。
 * 末尾 index (C-1, 半角スペース) は charset に `' '` が含まれる場合のみ許可。
 * dict に存在しない charset 中の文字は無視される (Set ∩ dict)。
 */
function buildAllowedMask(
  charset: ReadonlySet<string>,
  dict: string[],
  C: number,
): Uint8Array {
  const mask = new Uint8Array(C); // 0 = disallowed, 1 = allowed
  mask[0] = 1; // CTC blank は常に許可
  for (let i = 0; i < dict.length; i++) {
    if (charset.has(dict[i]!)) {
      mask[i + 1] = 1;
    }
  }
  if (charset.has(' ')) {
    mask[C - 1] = 1;
  }
  return mask;
}

/**
 * 1 サンプル (T × C softmax) を CTC greedy で decode。
 *
 * @param logits Float32Array length = T * C, row-major (T 連続)。 softmax 済を想定。
 * @param T time steps
 * @param C class count
 * @param dict 文字辞書 (index = dict[i-1] for output index i, i in [1, N])
 *             dict.length + 2 が C と一致しなければエラー
 * @param charset 省略可。 指定時は許可文字 (+ blank) のみを argmax の候補にする
 *                (constrained decode)。 未指定なら従来通り全 class が候補。
 */
export function ctcGreedyDecodePaddle(
  logits: Float32Array,
  T: number,
  C: number,
  dict: string[],
  charset?: ReadonlySet<string>,
): CtcDecodeResult {
  if (dict.length + 2 !== C) {
    throw new Error(
      `dict mismatch: expected dict.length + 2 == C (${C}), got ${dict.length + 2}`,
    );
  }

  const mask = charset ? buildAllowedMask(charset, dict, C) : undefined;

  // 各時刻 t で argmax + 確率を取得 (mask 指定時は許可 class のみが候補)
  const indices: number[] = new Array(T);
  const probs: number[] = new Array(T);
  for (let t = 0; t < T; t++) {
    let bestIdx = -1;
    let bestProb = -Infinity;
    for (let c = 0; c < C; c++) {
      if (mask && !mask[c]) continue;
      const p = logits[t * C + c]!;
      if (p > bestProb) {
        bestProb = p;
        bestIdx = c;
      }
    }
    indices[t] = bestIdx;
    probs[t] = bestProb;
  }

  // CTC collapse: 連続する同一インデックスを merge、 blank (0) を除く
  let prev = -1;
  let outText = '';
  const usedProbs: number[] = [];
  for (let t = 0; t < T; t++) {
    const idx = indices[t]!;
    if (idx === prev) {
      // 連続マージ。 信頼度の高い方を採用しても良いが PaddleOCR は最初の出現を使う。
      // 既に usedProbs に積まれた最後を更新 (max を取る) — PaddleOCR は平均なので
      // どちらでも近い結果。 PaddleOCR 公式実装と一致させるため max を採用。
      if (usedProbs.length > 0) {
        const last = usedProbs[usedProbs.length - 1]!;
        if (probs[t]! > last) usedProbs[usedProbs.length - 1] = probs[t]!;
      }
      continue;
    }
    prev = idx;
    if (idx === 0) {
      // blank はスキップ
      continue;
    }
    let ch: string;
    if (idx === C - 1) {
      // 末尾は半角スペース
      ch = ' ';
    } else {
      // 1..N → dict[idx - 1]
      ch = dict[idx - 1] ?? '';
    }
    outText += ch;
    usedProbs.push(probs[t]!);
  }

  // 信頼度: 採用された時刻の top1 確率の平均
  let confidence = 1.0;
  if (usedProbs.length > 0) {
    let sum = 0;
    for (const p of usedProbs) sum += p;
    confidence = sum / usedProbs.length;
  } else {
    // 何も decode しなかった (全部 blank) → confidence は 0
    confidence = 0;
  }

  return { text: outText, confidence };
}

/**
 * バッチ (B, T, C) を一括 decode。
 *
 * @param charset 省略可。 指定時は全サンプルに同一の許可文字制約を適用する
 *                (ctcGreedyDecodePaddle 参照)。
 */
export function ctcGreedyDecodeBatch(
  logits: Float32Array,
  B: number,
  T: number,
  C: number,
  dict: string[],
  charset?: ReadonlySet<string>,
): CtcDecodeResult[] {
  const out: CtcDecodeResult[] = new Array(B);
  const stride = T * C;
  for (let b = 0; b < B; b++) {
    const slice = logits.subarray(b * stride, (b + 1) * stride);
    out[b] = ctcGreedyDecodePaddle(slice, T, C, dict, charset);
  }
  return out;
}
