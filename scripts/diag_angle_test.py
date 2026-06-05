"""傾きが誤読原因かの統制実験 + ドメイン(コントラスト)差の確認。
1) 良好な Q1 crop を人工回転 → EM が落ちるか(角度の因果)
2) Q1 vs Q4 のコントラスト(gray std)比較(別ドメインか)
"""
import sys, re
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "packages/trainer/src")
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; BLANK = 36
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"; H, W = 32, 128

def decode(bgr):
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
    y = 0.2126*rgb[..., 0]+0.7152*rgb[..., 1]+0.0722*rgb[..., 2]
    x = ((y/255.0-0.5)/0.5).astype(np.float32)
    lg = sess.run([OUTP], {INP: x[None, None, :, :]})[0][0]
    arg = lg.argmax(-1); out = []; prev = -1
    for t in arg:
        if t != prev and t != BLANK: out.append(CHARSET[t])
        prev = t
    return "".join(out)

def rotate(bgr, deg):
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]

# Q1(タイト・高アスペクト)と Q4(四角い)を分ける
q1, q4 = [], []
for fn, gt in pos:
    bgr = cv2.imread(f"{base}/{fn}")
    if bgr is None: continue
    h, w = bgr.shape[:2]
    ar = w/h
    if ar >= 4.5:
        q1.append((fn, gt, bgr))
    elif ar < 2.3:
        q4.append((fn, gt, bgr))

import random; random.seed(0)
q1s = random.sample(q1, min(400, len(q1)))
print(f"Q1(ar>=4.5)={len(q1)} sample {len(q1s)} / Q4(ar<2.3)={len(q4)}")

# 1) Q1 を人工回転 → EM
print("\n=== 角度の因果(Q1良好cropを人工回転) ===")
for deg in [0, 8, 15, 22]:
    em = sum(decode(rotate(b, deg)) == gt for fn, gt, b in q1s)/len(q1s)*100
    print(f"  回転 {deg:>2}°: EM={em:.1f}%")

# 2) コントラスト比較
def contrast(crops):
    return np.mean([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).std() for _, _, b in crops])
print("\n=== コントラスト(gray std; 低いほど低コントラスト) ===")
print(f"  Q1(タイト): {contrast(q1s):.1f}")
print(f"  Q4(四角い): {contrast(q4[:400]):.1f}")
