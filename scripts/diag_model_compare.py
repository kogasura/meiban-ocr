"""(c) モデル選定: v6 fixed-head vs v7-crnn を full-frame (タイト単枚crop) で読字比較。"""
import sys, re, random, statistics as st
sys.path.insert(0, "packages/trainer/src")
import numpy as np, cv2, onnxruntime as ort
from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits, _detect_model_type_and_tokenizer, crop_and_normalize)


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


rows = [l.rstrip("\n").split("\t") for l in open("data/recognition/test/labels.tsv")][1:]
pos = [(r[0], r[1]) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]
random.seed(0)
sample = random.sample(pos, min(300, len(pos)))
base = "data/recognition"
print(f"positive test crops total={len(pos)} sampled={len(sample)}")

for name, onnx in [("v6 fixed-head (949KB)", "models/meiban-ocr-real-v6-rnn.fp32.onnx"),
                   ("v7-crnn ctc (16.5MB)", "models/meiban-ocr-real-v7-crnn.fp32.onnx")]:
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    mt, tok = _detect_model_type_and_tokenizer(sess)
    inp = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    em = 0; n = 0; confs = []; cc = []; cw = []
    for fn, gt in sample:
        img = cv2.imread(f"{base}/{fn}")
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x = crop_and_normalize(rgb, [0, 0, w, h]).astype(np.float32)[None, None, :, :]
        dec = _decode_logits(mt, tok, sess.run([outp], {inp: x})[0])
        t, c = dec[0]
        ok = norm(t) == norm(gt)
        em += ok; n += 1; confs.append(c)
        (cc if ok else cw).append(c)
    print(f"\n=== {name}  model_type={mt} ===")
    print(f"  full-frame EM: {em}/{n} = {em/n*100:.1f}%")
    print(f"  conf median: 全体={st.median(confs):.3f} "
          f"正解={st.median(cc) if cc else 0:.3f} 誤={st.median(cw) if cw else 0:.3f}")
    print(f"  誤読で conf>=0.5 (=accept されFP化): {sum(1 for c in cw if c>=0.5)}/{len(cw)}")
