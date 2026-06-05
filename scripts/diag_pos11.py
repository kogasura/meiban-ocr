"""pos11(末尾桁)誤りの機序切り分け: 未知tail汎化失敗 vs crop端で切れ。
clean baseline(runs/20260604-172318)を held-out test に当てる。"""
import sys, re, random
from collections import Counter
import numpy as np, cv2, torch
sys.path.insert(0, "packages/trainer/src")
from meiban_ocr_trainer.models.crnn_pretrained import CRNNPretrained
from meiban_ocr_trainer.tools.diagnose_pipeline import crop_and_normalize

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; BLANK = 36
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
base = "data/recognition"

ck = torch.load("runs/20260604-172318/best.pt", map_location="cpu", weights_only=False)
model = CRNNPretrained(num_classes=37, hidden_size=256)
model.load_state_dict(ck["model_state"]); model = model.to(DEV).eval()

@torch.no_grad()
def decode(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    x = crop_and_normalize(rgb, [0, 0, w, h]).astype(np.float32)[None, None, :, :]
    lg = model(torch.from_numpy(x).to(DEV)).cpu().numpy()[0]
    arg = lg.argmax(-1); out = []; prev = -1
    for t in arg:
        if t != prev and t != BLANK: out.append(CHARSET[t]); prev = t
        else: prev = t
    return "".join(out)

rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/test/labels.tsv")] if False else None
# labels_serial_split.tsv の test positive
recs = []
for l in open(f"{base}/labels_serial_split.tsv"):
    p = l.rstrip("\n").split("\t")
    if len(p) > 5 and p[2] == "test" and p[5] == "positive" and p[1]:
        recs.append((p[0], norm(p[1])))

# 訓練 split の末尾桁頻度(バイアス確認)
train_last = Counter()
for l in open(f"{base}/labels_serial_split.tsv"):
    p = l.rstrip("\n").split("\t")
    if len(p) > 5 and p[2] == "train" and p[5] == "positive" and p[1]:
        s = norm(p[1])
        if len(s) == 12: train_last[s[11]] += 1

last_conf = Counter()      # gt_last -> pred_last
pos11_only = []            # (fn, gt, pred) で pos11 のみ誤り
edge_ink_err, edge_ink_ok = [], []
n11 = ntot = 0
for fn, gt in recs:
    bgr = cv2.imread(f"{base}/{fn}")
    if bgr is None or len(gt) != 12: continue
    pred = decode(bgr); ntot += 1
    # 右端10%のインク量(暗textの割合)= 最終桁が端に寄ってるか
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rcol = g[:, int(g.shape[1]*0.9):]
    ink = (rcol < (g.mean()-10)).mean()
    if len(pred) == 12:
        diffs = [i for i in range(12) if pred[i] != gt[i]]
        if diffs == [11]:
            n11 += 1; pos11_only.append((fn, gt, pred))
            last_conf[f"{gt[11]}->{pred[11]}"] += 1
            edge_ink_err.append(ink)
        elif not diffs:
            edge_ink_ok.append(ink)

print(f"held-out test positive(12桁)={ntot}")
print(f"pos11のみ誤り = {n11}")
print(f"末尾桁 混同 top: {last_conf.most_common(10)}")
print(f"訓練の末尾桁頻度: {dict(sorted(train_last.items()))}")
pred_last_dist = Counter(p.split('->')[1] for p in last_conf.elements())
print(f"誤読時の pred末尾桁 分布: {dict(sorted(pred_last_dist.items()))}")
import statistics as st
if edge_ink_err and edge_ink_ok:
    print(f"右端インク率 中央値: 誤り={st.median(edge_ink_err):.3f}  正解={st.median(edge_ink_ok):.3f}")

# montage(最大16枚)
random.seed(0); sample = random.sample(pos11_only, min(16, len(pos11_only)))
tiles = []
for fn, gt, pred in sample:
    b = cv2.imread(f"{base}/{fn}"); h, w = b.shape[:2]
    t = cv2.resize(b, (320, max(1, int(h*320/w))))
    bar = np.full((20, 320, 3), 30, np.uint8)
    cv2.putText(bar, f"gt={gt} pred={pred}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(np.vstack([bar, t, np.zeros((3, 320, 3), np.uint8)]))
if tiles:
    cv2.imwrite("/tmp/pos11_montage.png", np.vstack(tiles))
    print("saved /tmp/pos11_montage.png")
