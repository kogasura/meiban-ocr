"""(b) 仮説検証: 製造番号プレフィックスが認識を阻害しているか。
白ラベル領域 → 行検出 → 各行の[全体/右60%/右45%]を認識して GT 回収を比較。"""
import sys, re, json
sys.path.insert(0, "packages/trainer/src")
import numpy as np, cv2
import onnxruntime as ort
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer, crop_and_normalize)
from meiban_ocr_trainer.vendors import ERICSSON
STRICT = ERICSSON.strict_regex
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
sess = ort.InferenceSession("models/meiban-ocr-real-v7-crnn.fp32.onnx", providers=["CPUExecutionProvider"])
MT, TOK = _detect_model_type_and_tokenizer(sess)
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name
gt = set(json.load(open("runs/session_eval/10_44.json"))["truth"])

def rec(rgb, box):
    crop = crop_and_normalize(rgb, box).astype(np.float32)[None, None, :, :]
    return _decode_logits(MT, TOK, sess.run([OUTP], {INP: crop})[0])[0]

def detect_lines(bgr):
    """白ラベル上の text-line を検出(細かめ close で行を分離)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh2 = cv2.boundingRect(c)
        ar = bw / max(1, bh2)
        if 3 <= ar <= 14 and bw >= 50 and 8 <= bh2 <= 60 and bw <= w * 0.5:
            boxes.append([x, y, x + bw, y + bh2])
    return boxes

cap = cv2.VideoCapture("videos/10_44.mov"); fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(15.0 * fps)); ok, bgr = cap.read(); cap.release()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
lines = detect_lines(bgr)
print(f"t=15s: 検出 text-line = {len(lines)}")

variants = {"行全体": lambda b: b,
            "右60%": lambda b: [int(b[0]+(b[2]-b[0])*0.40), b[1], b[2], b[3]],
            "右45%": lambda b: [int(b[0]+(b[2]-b[0])*0.55), b[1], b[2], b[3]]}
for vname, fn in variants.items():
    found = set(); samples = []
    for b in lines:
        bb = fn(b)
        if bb[2]-bb[0] < 20: continue
        t, c = rec(rgb, bb)
        nt = norm(t)
        if STRICT.match(nt):
            if nt in gt and c >= 0.5: found.add(nt)
            if len(samples) < 6: samples.append((nt, round(c, 2), nt in gt))
    print(f"  [{vname}] GT回収(conf>=0.5)={len(found)}/{len(gt)}  例:{samples}")
