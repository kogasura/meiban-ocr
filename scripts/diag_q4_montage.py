"""Q4(低アスペクト=四角い)かつ stretch で誤読の crop を実際に見る。
なぜ四角いのか(角度 / 余白 / 複数行)を目視で判定するための montage。"""
import sys, re
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "packages/trainer/src")
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; BLANK = 36
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"; H, W = 32, 128

def decode_stretch(bgr):
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
    y = 0.2126*rgb[..., 0]+0.7152*rgb[..., 1]+0.0722*rgb[..., 2]
    x = ((y/255.0-0.5)/0.5).astype(np.float32)
    logits = sess.run([OUTP], {INP: x[None, None, :, :]})[0][0]
    arg = logits.argmax(-1); out = []; prev = -1
    for t in arg:
        if t != prev and t != BLANK: out.append(CHARSET[t])
        prev = t
    return "".join(out)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]

picks = []
for fn, gt in pos:
    bgr = cv2.imread(f"{base}/{fn}")
    if bgr is None: continue
    h, w = bgr.shape[:2]
    ar = w/h
    if ar < 2.3:  # 四角い crop
        pred = decode_stretch(bgr)
        if pred != gt:
            picks.append((fn, gt, pred, h, w, ar, bgr))
    if len(picks) >= 12: break

print(f"四角い(w/h<2.3)誤読 crop: {len(picks)} 件を montage 化")
# montage: 各 crop を幅320に揃えて縦積み、注釈付き
tiles = []
for fn, gt, pred, h, w, ar, bgr in picks:
    scale = 320/w
    t = cv2.resize(bgr, (320, max(1, int(h*scale))))
    bar = np.full((22, 320, 3), 30, np.uint8)
    cv2.putText(bar, f"gt={gt} pred={pred} {w}x{h} ar={ar:.2f}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    tiles.append(np.vstack([bar, t, np.full((4, 320, 3), 0, np.uint8)]))
    print(f"  {fn}: gt={gt} pred={pred} {w}x{h} ar={ar:.2f}")
montage = np.vstack(tiles)
cv2.imwrite("/tmp/q4_montage.png", montage)
print("saved /tmp/q4_montage.png", montage.shape)
