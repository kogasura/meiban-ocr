"""deskew が Q4(傾いた四角い crop)を救うか検証。
minAreaRect で text 角度推定 → 水平化 → tight再crop → stretch → 認識。"""
import sys, re, random
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

def deskew(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # text を前景に(暗/明 両対応で Otsu + 多い方を背景とみなし反転)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.5: th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 20: return bgr
    rect = cv2.minAreaRect(pts)
    ang = rect[-1]
    if ang < -45: ang += 90
    if ang > 45: ang -= 90
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    rot = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # 回転後の text bbox で tight 再crop
    rth = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_NEAREST)
    p2 = cv2.findNonZero(rth)
    if p2 is None: return rot
    x, y, bw, bh = cv2.boundingRect(p2)
    pad = int(bh*0.15)
    x0, y0 = max(0, x-pad), max(0, y-pad); x1, y1 = min(w, x+bw+pad), min(h, y+bh+pad)
    return rot[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else rot

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]
q4 = []
for fn, gt in pos:
    bgr = cv2.imread(f"{base}/{fn}")
    if bgr is None: continue
    h, w = bgr.shape[:2]
    if w/h < 2.3: q4.append((fn, gt, bgr))
random.seed(0); q4s = random.sample(q4, min(500, len(q4)))
print(f"Q4(四角い)={len(q4)} sample {len(q4s)}")
base_em = sum(decode(b) == gt for _, gt, b in q4s)/len(q4s)*100
desk_em = sum(decode(deskew(b)) == gt for _, gt, b in q4s)/len(q4s)*100
print(f"  stretch そのまま : EM={base_em:.1f}%")
print(f"  deskew+tight再crop: EM={desk_em:.1f}%  Δ={desk_em-base_em:+.1f}")
