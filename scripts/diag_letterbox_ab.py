"""前処理 A/B 検証: stretch(現状) vs letterbox(アスペクト維持+白padding)。
height 四分位別に EM を比較し、緩い枠(Q4)が letterbox で戻るか確認。
注: モデルは stretch 訓練済→letterboxは推論ミスマッチ。それでも改善するなら歪みが主因。
"""
import sys, re
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "packages/trainer/src")
from meiban_ocr_trainer.tools.diagnose_pipeline import _detect_model_type_and_tokenizer

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; BLANK = 36
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"
H, W = 32, 128

def to_gray_norm(bgr_canvas):
    rgb = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    y = 0.2126*rgb[..., 0] + 0.7152*rgb[..., 1] + 0.0722*rgb[..., 2]
    return ((y/255.0 - 0.5)/0.5).astype(np.float32)

def prep_stretch(bgr):
    return to_gray_norm(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA))

def prep_letterbox(bgr, pad=255):
    h, w = bgr.shape[:2]
    s = min(H/h, W/w); nh, nw = max(1, round(h*s)), max(1, round(w*s))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), pad, np.uint8)
    y0, x0 = (H-nh)//2, (W-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return to_gray_norm(canvas)

def decode(x):
    logits = sess.run([OUTP], {INP: x[None, None, :, :]})[0][0]
    arg = logits.argmax(-1); out = []; prev = -1
    for t in arg:
        if t != prev and t != BLANK: out.append(CHARSET[t])
        prev = t
    return "".join(out)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]

recs = []  # (h, w, ok_stretch, ok_letterbox)
for fn, gt in pos:
    bgr = cv2.imread(f"{base}/{fn}")
    if bgr is None: continue
    h, w = bgr.shape[:2]
    ok_s = decode(prep_stretch(bgr)) == gt
    ok_l = decode(prep_letterbox(bgr)) == gt
    recs.append((h, w, ok_s, ok_l))

N = len(recs)
ems = sum(r[2] for r in recs)/N*100
eml = sum(r[3] for r in recs)/N*100
print(f"n={N}")
print(f"EM stretch(現状) = {ems:.1f}%")
print(f"EM letterbox     = {eml:.1f}%")

# height 四分位別
hs = np.array([r[0] for r in recs])
qs = np.quantile(hs, [0.25, 0.5, 0.75])
def q(h): return 0 if h < qs[0] else (1 if h < qs[1] else (2 if h < qs[2] else 3))
print(f"\nheight四分位 境界: {qs.round(0)}")
for qi in range(4):
    sub = [r for r in recs if q(r[0]) == qi]
    if not sub: continue
    s = sum(r[2] for r in sub)/len(sub)*100
    l = sum(r[3] for r in sub)/len(sub)*100
    ar = np.median([r[1]/r[0] for r in sub])
    print(f"  Q{qi+1} (n={len(sub)}, 中央アスペクトw/h={ar:.2f}): stretch={s:.1f}%  letterbox={l:.1f}%  Δ={l-s:+.1f}")
