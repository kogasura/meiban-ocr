"""E2E 成功率の細分化: 躓き箇所を (画像品質 / 特徴曖昧 / デコード取りこぼし) に分解。

test positive crop(=完璧に枠取りされた reticle 相当)で:
  - 全体 EM
  - 画像品質(ブレ=Laplacian分散 / crop高さ)四分位ごとの EM  -> 解像度/前処理レバーか
  - 誤読の位置別 top-2 マージン  -> 特徴が曖昧(near-miss)か 確信誤り(feature/data)か
  - 「1文字違い かつ gt が top-2」= より良いデコード/fixed-head 制約で救える上限
"""
import sys, re
from collections import Counter, defaultdict
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "packages/trainer/src")
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _detect_model_type_and_tokenizer, crop_and_normalize)

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 36
C2I = {c: i for i, c in enumerate(CHARSET)}
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def ctc_greedy_with_probs(logits):
    """(T,37) -> list[(emitted_char, probvec)] CTC greedy collapse。"""
    probs = softmax(logits)
    arg = probs.argmax(-1)
    out = []; prev = -1
    for t in range(len(arg)):
        idx = arg[t]
        if idx != prev and idx != BLANK:
            out.append(probs[t])
        prev = idx
    return out

def analyze(fn):
    img = cv2.imread(f"{base}/{fn}")
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    h, w = gray.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = crop_and_normalize(rgb, [0, 0, w, h]).astype(np.float32)[None, None, :, :]
    logits = sess.run([OUTP], {INP: x})[0][0]  # (T,37)
    emitted = ctc_greedy_with_probs(logits)
    pred = "".join(CHARSET[pv.argmax()] for pv in emitted)
    return pred, emitted, blur, h, w

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]

recs = []
pos_margin = defaultdict(list)   # position -> list of (correct?, p_top1, p_gt)
recoverable_top2 = 0; oneoff = 0; gt_not_in_top2 = 0
for fn, gt in pos:
    r = analyze(fn)
    if r is None: continue
    pred, emitted, blur, h, w = r
    ok = pred == gt
    recs.append((ok, blur, h, w))
    if not ok and len(pred) == len(gt):
        diffs = [i for i in range(len(gt)) if pred[i] != gt[i]]
        if len(diffs) == 1:
            oneoff += 1
            i = diffs[0]; pv = emitted[i]
            order = pv.argsort()[::-1]
            top2_idx = [j for j in order if j != BLANK][:2]
            gtid = C2I[gt[i]]
            if gtid in top2_idx:
                recoverable_top2 += 1
            else:
                gt_not_in_top2 += 1
            pos_margin[i].append((pv.argmax() == gtid, float(pv.max()), float(pv[gtid])))

N = len(recs); em = sum(1 for r in recs if r[0])
print(f"=== E2E(test crop, reticle相当) n={N}  EM={em/N*100:.1f}% ===")

# 画像品質四分位ごとの EM
import numpy as np
def bucket_em(key_idx, label):
    vals = sorted(set(r[key_idx] for r in recs))
    qs = np.quantile([r[key_idx] for r in recs], [0.25, 0.5, 0.75])
    buckets = {f"Q1(<{qs[0]:.0f})": [], f"Q2": [], f"Q3": [], f"Q4(>{qs[2]:.0f})": []}
    for r in recs:
        v = r[key_idx]
        b = "Q1(<%.0f)" % qs[0] if v < qs[0] else ("Q2" if v < qs[1] else ("Q3" if v < qs[2] else "Q4(>%.0f)" % qs[2]))
        buckets[b].append(r[0])
    print(f"\n  --- {label} 四分位ごとの EM ---")
    for b, lst in buckets.items():
        if lst: print(f"    {b}: EM={sum(lst)/len(lst)*100:.1f}%  (n={len(lst)})")
bucket_em(1, "ブレ(Laplacian分散; 高いほど鮮明)")
bucket_em(2, "crop高さ(px; 解像度)")

print(f"\n=== 誤読の救済可能性(1文字違いケース) ===")
print(f"  1文字違いの誤読: {oneoff}")
print(f"  うち gt が top-2 に居る(=良いデコード/制約で救える): {recoverable_top2}")
print(f"  gt が top-2 にすら無い(=特徴が弱い/data不足): {gt_not_in_top2}")

print(f"\n=== 誤読位置別の確信度(top1 vs gt の確率) ===")
for i in sorted(pos_margin):
    rows_i = pos_margin[i]
    if len(rows_i) < 5: continue
    p_top1 = np.mean([r[1] for r in rows_i]); p_gt = np.mean([r[2] for r in rows_i])
    print(f"  pos{i}: 誤り{len(rows_i)}件  平均 p(top1)={p_top1:.2f}  平均 p(gt)={p_gt:.2f}  margin={p_top1-p_gt:.2f}")
