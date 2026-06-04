"""session の「発火ゼロ」原因切り分け: どの段(窓/認識conf/パターン)で全滅するか。

各動画の先頭数フレームで sliding-window + recognizer を回し、フィルタ前の
生 (text, conf) を集計する。
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "packages/trainer/src")
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer,
    crop_and_normalize, generate_windows,
)
from meiban_ocr_trainer.vendors import ERICSSON
import cv2

STRICT = ERICSSON.strict_regex
PARTIAL = ERICSSON.partial_regex
SCALES = [0.7, 1.0, 1.4]
CONF_T = 0.7
ONNX = "models/meiban-ocr-real-v6-rnn.fp32.onnx"

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
model_type, tok = _detect_model_type_and_tokenizer(sess)
inp = sess.get_inputs()[0].name
outp = sess.get_outputs()[0].name
print(f"model_type={model_type}  onnx={ONNX}  scales={SCALES}  conf_T={CONF_T}\n")


def sample_frames(path, n=3, fps=3.3):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(src / fps)))
    out, idx = [], 0
    while len(out) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            out.append(bgr)
        idx += 1
    cap.release()
    return out


def analyze(video, n=3):
    print(f"==================== {video} ====================")
    frames = sample_frames(video, n)
    for fi, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        wins = generate_windows(w, h, {"scales": SCALES})
        confs, strict_hits, partial_hits, pass_conf = [], 0, 0, 0
        top = []  # (conf, text)
        for i in range(0, len(wins), 64):
            batch = wins[i:i+64]
            crops = np.stack([crop_and_normalize(rgb, b) for b in batch])
            x = crops[:, None, :, :]
            logits = sess.run([outp], {inp: x})[0]
            for (text, conf) in _decode_logits(model_type, tok, logits):
                confs.append(conf)
                if conf >= CONF_T:
                    pass_conf += 1
                if text and STRICT.match(text):
                    strict_hits += 1
                    top.append((conf, text))
                elif text and PARTIAL.search(text):
                    partial_hits += 1
                    top.append((conf, text))
        confs = np.array(confs) if confs else np.array([0.0])
        top.sort(reverse=True)
        print(f"[frame {fi}] {w}x{h}  windows={len(wins)}")
        print(f"   conf: max={confs.max():.3f} p99={np.percentile(confs,99):.3f} "
              f"p90={np.percentile(confs,90):.3f} median={np.median(confs):.3f}")
        print(f"   conf>={CONF_T}: {pass_conf}/{len(confs)}  "
              f"strict-pattern hits={strict_hits}  partial hits={partial_hits}")
        print(f"   top pattern preds: {top[:5]}")
    print()


analyze("videos/9_47-5.mov")   # 37.5% (発火する)
analyze("videos/10_44.mov")    # 0%   (発火しない)
