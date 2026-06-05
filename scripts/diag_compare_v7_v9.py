"""v7(旧aug ±2°) vs v9-aug(±12°) を torch直接で比較。
回転耐性 / 全EM / 5→6混同 / Q4(四角い) を同条件で測る。"""
import sys, re, random
import numpy as np, cv2, torch
sys.path.insert(0, "packages/trainer/src")
from meiban_ocr_trainer.models.crnn_pretrained import CRNNPretrained

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; BLANK = 36
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
base = "data/recognition"; H, W = 32, 128

def load_model(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = CRNNPretrained(num_classes=37, hidden_size=256)
    m.load_state_dict(c["model_state"])
    return m.to(DEV).eval()

def prep(bgr):
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
    y = 0.2126*rgb[..., 0]+0.7152*rgb[..., 1]+0.0722*rgb[..., 2]
    return ((y/255.0-0.5)/0.5).astype(np.float32)

@torch.no_grad()
def decode_batch(model, arrs):
    x = torch.from_numpy(np.stack(arrs)[:, None, :, :]).to(DEV)
    logits = model(x).cpu().numpy()  # (B,T,37)
    outs = []
    for lg in logits:
        arg = lg.argmax(-1); s = []; prev = -1
        for t in arg:
            if t != prev and t != BLANK: s.append(CHARSET[t])
            prev = t
        outs.append("".join(s))
    return outs

def rotate(bgr, deg):
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")][1:]
pos = [(r[0], norm(r[1])) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]

# crop を一括ロード
data = []
for fn, gt in pos:
    b = cv2.imread(f"{base}/{fn}")
    if b is not None: data.append((b, gt, b.shape[1]/b.shape[0]))
print(f"test positive crops: {len(data)}  device={DEV}")

def run_em(model, imgs_gts, batch=256):
    ok = 0
    for i in range(0, len(imgs_gts), batch):
        chunk = imgs_gts[i:i+batch]
        preds = decode_batch(model, [prep(b) for b, _ in chunk])
        ok += sum(p == g for p, (_, g) in zip(preds, chunk))
    return ok/len(imgs_gts)*100

def count_5to6(model, imgs_gts, batch=256):
    c = 0
    for i in range(0, len(imgs_gts), batch):
        chunk = imgs_gts[i:i+batch]
        preds = decode_batch(model, [prep(b) for b, _ in chunk])
        for p, (_, g) in zip(preds, chunk):
            if len(p) == len(g):
                for a, b2 in zip(g, p):
                    if a == "5" and b2 == "6": c += 1
    return c

for name, ck in [("v7 (旧aug±2°)", "runs/20260602-204153/best.pt"),
                 ("v9 (新aug±12°)", "runs/20260604-150048/best.pt")]:
    m = load_model(ck)
    ig = [(b, g) for b, g, ar in data]
    em_all = run_em(m, ig)
    n56 = count_5to6(m, ig)
    q4 = [(b, g) for b, g, ar in data if ar < 2.3]
    q1 = [(b, g) for b, g, ar in data if ar >= 4.5]
    em_q4 = run_em(m, q4); em_q1 = run_em(m, q1)
    # 回転耐性(Q1を人工回転)
    random.seed(0); q1s = random.sample(q1, min(400, len(q1)))
    rot = {}
    for d in [0, 8, 15]:
        rot[d] = run_em(m, [(rotate(b, d), g) for b, g in q1s])
    print(f"\n=== {name} ===")
    print(f"  全EM={em_all:.1f}%  Q1(タイト)={em_q1:.1f}%  Q4(四角い)={em_q4:.1f}%  5→6混同={n56}")
    print(f"  回転耐性(Q1): 0°={rot[0]:.1f}%  8°={rot[8]:.1f}%  15°={rot[15]:.1f}%")
