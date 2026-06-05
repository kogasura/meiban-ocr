"""5→6 誤読が systematic(consensus で直らない)か random(直る)か。
5→6 で誤った crop に微小ジッタを与え、予測が正解へ flip するか測る。"""
import sys, re, random
sys.path.insert(0, "packages/trainer/src")
import numpy as np, cv2, onnxruntime as ort
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer, crop_and_normalize)
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
MT, TOK = _detect_model_type_and_tokenizer(sess)
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
base = "data/recognition"

def pred_rgb(rgb):
    h, w = rgb.shape[:2]
    x = crop_and_normalize(rgb, [0, 0, w, h]).astype(np.float32)[None, None, :, :]
    t, c = _decode_logits(MT, TOK, sess.run([OUTP], {INP: x})[0])[0]
    return norm(t), c

def jitter(img, i):
    h, w = img.shape[:2]
    rng = random.Random(i)
    # 明度
    a = 0.8 + rng.random()*0.4; b = rng.randint(-20, 20)
    out = cv2.convertScaleAbs(img, alpha=a, beta=b)
    # 微小アフィン(シフト+スケール+回転)
    ang = rng.uniform(-3, 3); sc = 0.95 + rng.random()*0.1
    tx = rng.uniform(-0.04, 0.04)*w; ty = rng.uniform(-0.04, 0.04)*h
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, sc); M[0, 2] += tx; M[1, 2] += ty
    return cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]
random.seed(1); random.shuffle(pos)

# 5→6 誤読の crop を集める(最大25件)
cases = []
for fn, gt in pos:
    img = cv2.imread(f"{base}/{fn}")
    if img is None: continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p, _ = pred_rgb(rgb)
    # gt と1文字違いで、その違いが 5->6
    if len(p) == len(gt) and p != gt:
        diffs = [(a, b) for a, b in zip(gt, p) if a != b]
        if diffs == [("5", "6")]:
            cases.append((fn, gt, rgb))
    if len(cases) >= 25: break

print(f"5→6 単一誤読 crop: {len(cases)} 件で各12ジッタ試行")
systematic = 0; flips = []
for fn, gt, rgb in cases:
    n_correct = 0
    for i in range(12):
        p, _ = pred_rgb(jitter(rgb, i))
        if p == gt: n_correct += 1
    flips.append(n_correct)
    if n_correct == 0: systematic += 1
print(f"  ジッタ12回中の正解回数 分布: {sorted(flips)}")
print(f"  完全 systematic(12回とも誤読のまま) = {systematic}/{len(cases)}")
print(f"  平均 flip-to-correct 率 = {np.mean([f/12 for f in flips])*100:.1f}%")
