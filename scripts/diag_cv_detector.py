"""(b) 古典CV検出器の試作。

トレイフレームから OpenCV の morphological text-line 抽出で銘板矩形候補を出し、
各候補を v7-crnn で full-frame 認識して GT 回収率を測る。
sliding-window(全画面ブルート)の代替が成立するかを検証。
"""
import sys, re, json
sys.path.insert(0, "packages/trainer/src")
import numpy as np, cv2, onnxruntime as ort
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer, crop_and_normalize)
from meiban_ocr_trainer.vendors import ERICSSON

STRICT = ERICSSON.strict_regex
ONNX = "models/meiban-ocr-real-v7-crnn.fp32.onnx"
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
MT, TOK = _detect_model_type_and_tokenizer(sess)
INP, OUTP = sess.get_inputs()[0].name, sess.get_outputs()[0].name


def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def detect_textlines(bgr):
    """morphological gradient + 横長 close で text-line 候補矩形を返す。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 1) blackhat: 明るい背景上の暗い刻印/印字を強調
    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)
    # 2) 2値化
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 3) 横方向に close して文字を1行に連結
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh2 = cv2.boundingRect(c)
        ar = bw / max(1, bh2)
        area = bw * bh2
        # 銘板シリアルの見た目: 横長(2.5〜12:1), ある程度の面積, 極端に小/大を除外
        if 2.5 <= ar <= 12 and bw >= 40 and bh2 >= 10 and area >= 800 and bw <= w * 0.6:
            pad = int(bh2 * 0.25)
            boxes.append([max(0, x - pad), max(0, y - pad),
                          min(w, x + bw + pad), min(h, y + bh2 + pad)])
    return boxes


def recognize(rgb, box):
    crop = crop_and_normalize(rgb, box).astype(np.float32)[None, None, :, :]
    t, c = _decode_logits(MT, TOK, sess.run([OUTP], {INP: crop})[0])[0]
    return t, c


def frame_at(path, ts_sec):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts_sec * fps))
    ok, b = cap.read()
    cap.release()
    return b if ok else None


gt = set(json.load(open("runs/session_eval/10_44.json"))["truth"])
print(f"GT(10_44) = {len(gt)} 枚")

# sliding-window ベスト付近のフレームで評価
for ts in [14.0, 15.0, 16.0]:
    bgr = frame_at("videos/10_44.mov", ts)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_textlines(bgr)
    found, hi = set(), set()
    vis = bgr.copy()
    for box in boxes:
        t, c = recognize(rgb, box)
        nt = norm(t)
        color = (120, 120, 120)
        if STRICT.match(nt):
            color = (0, 200, 0) if nt in gt else (0, 140, 255)
            if nt in gt:
                found.add(nt)
                if c >= 0.5:
                    hi.add(nt)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
    cv2.imwrite(f"/tmp/cv_det_10_44_{int(ts)}s.png", vis)
    print(f"\nt={ts}s: 候補矩形={len(boxes)}  GT回収={len(found)}/{len(gt)}  "
          f"(conf>=0.5: {len(hi)})  -> /tmp/cv_det_10_44_{int(ts)}s.png")
