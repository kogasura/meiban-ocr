"""次の精度改善のボトルネック特定: v7-crnn の誤りパターンを全 test crop で解析。
A) positive: EM / CER / 誤り文字数分布 / 位置別誤り / 文字混同ペア
B) negative(real_neg): パターン有効出力率(=誤発火=FP リスク)
"""
import sys, re
from collections import Counter, defaultdict
sys.path.insert(0, "packages/trainer/src")
import numpy as np, cv2, onnxruntime as ort
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer, crop_and_normalize)
from meiban_ocr_trainer.vendors import ERICSSON
STRICT = ERICSSON.strict_regex
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())

sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
MT, TOK = _detect_model_type_and_tokenizer(sess)
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"

def lev(a, b):
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+(ca != cb))
            prev = cur
    return dp[-1]

def predict(fn):
    img = cv2.imread(f"{base}/{fn}")
    if img is None: return None, 0.0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    x = crop_and_normalize(rgb, [0, 0, w, h]).astype(np.float32)[None, None, :, :]
    t, c = _decode_logits(MT, TOK, sess.run([OUTP], {INP: x})[0])[0]
    return norm(t), c

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]
neg = [r[0] for r in rows if len(r) > 5 and r[5] != "positive"]

# ---------- A) positive ----------
em = tot = 0; cer_num = cer_den = 0
errmult = Counter(); pos_err = Counter(); conf_pairs = Counter()
conf_correct = []; conf_wrong = []
for fn, gt in pos:
    pred, c = predict(fn)
    if pred is None: continue
    tot += 1
    d = lev(pred, gt)
    cer_num += d; cer_den += len(gt)
    if pred == gt:
        em += 1; conf_correct.append(c)
    else:
        conf_wrong.append(c)
        errmult[min(d, 4)] += 1
        if len(pred) == len(gt):  # 同長なら位置別 substitution
            for i, (a, b) in enumerate(zip(gt, pred)):
                if a != b:
                    pos_err[i] += 1
                    conf_pairs[f"{a}->{b}"] += 1
print(f"=== A) POSITIVE (n={tot}) ===")
print(f"  EM = {em/tot*100:.1f}%   CER = {cer_num/cer_den*100:.2f}%")
import statistics as st
print(f"  conf median: 正解={st.median(conf_correct):.3f} 誤={st.median(conf_wrong) if conf_wrong else 0:.3f}")
print(f"  誤り文字数分布(edit dist): " + " ".join(f"{k}文字:{v}" for k, v in sorted(errmult.items())))
print(f"  位置別誤り(0-indexed, E[0]3[1]2[2]5[3]M[4]M[5]…): {dict(sorted(pos_err.items()))}")
print(f"  文字混同 top10: {conf_pairs.most_common(10)}")

# ---------- B) negative ----------
nfa = 0; nseen = 0; neg_examples = []
for fn in neg:
    pred, c = predict(fn)
    if pred is None: continue
    nseen += 1
    if STRICT.match(pred):           # negativeなのにパターン有効 = 誤発火
        if c >= 0.5:
            nfa += 1
            if len(neg_examples) < 8: neg_examples.append((pred, round(c, 2)))
print(f"\n=== B) NEGATIVE (real_neg, n={nseen}) ===")
print(f"  誤発火(pattern一致 & conf>=0.5) = {nfa}/{nseen} = {nfa/nseen*100:.2f}%")
print(f"  例: {neg_examples}")
