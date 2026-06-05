"""学習データの前提監査: 構成 / serial多様性 / train-test リーク / 数字分布。"""
import re
from collections import Counter
def norm(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
base = "data/recognition"

def load(split):
    rows = [l.rstrip("\n").split("\t") for l in open(f"{base}/{split}/labels.tsv")][1:]
    return rows

for split in ["train", "val", "test"]:
    rows = load(split)
    cat = Counter(r[5] if len(r) > 5 else "?" for r in rows)
    # source 接頭(real/replaced/synthetic を dir から)
    srcdir = Counter(r[0].split("/")[1] if "/" in r[0] else "?" for r in rows)
    pos = [norm(r[1]) for r in rows if len(r) > 5 and r[5] == "positive" and r[1]]
    uniq = set(pos)
    print(f"=== {split}: {len(rows)} 行 | category={dict(cat)} | dir={dict(srcdir)} | positive={len(pos)} uniqueシリアル={len(uniq)}")

# train-test リーク
tr = set(norm(r[1]) for r in load("train") if len(r) > 5 and r[5] == "positive" and r[1])
te_rows = [norm(r[1]) for r in load("test") if len(r) > 5 and r[5] == "positive" and r[1]]
te = set(te_rows)
overlap = tr & te
print(f"\n=== train∩test シリアル重複(リーク) ===")
print(f"  train unique={len(tr)} test unique={len(te)} 重複={len(overlap)} ({len(overlap)/len(te)*100:.1f}% of test unique)")
te_inst_leak = sum(1 for s in te_rows if s in tr)
print(f"  test インスタンスのうち train に同一シリアルが居る割合 = {te_inst_leak}/{len(te_rows)} = {te_inst_leak/len(te_rows)*100:.1f}%")

# 数字分布(train positive、 全位置 & エラー集中位置 3,10,11)
def digit_dist(serials, positions):
    out = {}
    for p in positions:
        c = Counter(s[p] for s in serials if len(s) > p)
        out[p] = {d: c.get(d, 0) for d in "0123456789"}
    return out
trp = [s for s in (norm(r[1]) for r in load("train") if len(r) > 5 and r[5]=="positive" and r[1]) if re.match(r"^E[39]\d{2}MM\d{6}$", s)]
print(f"\n=== train positive(厳格パターン一致 {len(trp)}件)の数字分布 ===")
allpos = Counter()
for s in trp:
    for ch in s:
        if ch.isdigit(): allpos[ch]+=1
print(f"  全数字頻度: {dict(sorted(allpos.items()))}")
dd = digit_dist(trp, [3, 9, 10, 11])
for p, c in dd.items():
    print(f"  position {p}: {c}")
