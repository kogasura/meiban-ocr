"""統合・再利用可能な認識評価ハーネス。

使い捨ての scripts/diag_error_analysis.py / diag_angle_test.py / diag_compare_v7_v9.py
を1つに統合。serial-disjoint な held-out test 上でモデルを多角的に評価する。

指標: EM / CER / 誤り文字数分布 / 位置別誤り / 文字混同top / 5→6件数 /
      正解・誤り median conf / アスペクト四分位別EM / 回転耐性(0/8/15°) / negative誤発火率

使用例:
    # v7 を serial-disjoint test で評価 + リーク版と並記
    python -m meiban_ocr_trainer.tools.eval_recognition \
        --model runs/20260602-204153/best.pt --compare-leaky

    # 複数モデル横並び
    python -m meiban_ocr_trainer.tools.eval_recognition \
        --models runs/20260602-204153/best.pt runs/20260604-150048/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from pathlib import Path

import cv2
import numpy as np

from meiban_ocr_trainer.tools.diagnose_pipeline import (
    _decode_logits,
    _detect_model_type_and_tokenizer,
    crop_and_normalize,
)
from meiban_ocr_trainer.tokenizer import CTCTokenizer
from meiban_ocr_trainer.vendors import ERICSSON

STRICT = ERICSSON.strict_regex


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _lev(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


# ----- model backends -----
def load_predictor(model_path: Path, resize_mode_override: str | None = None):
    """returns (predict_fn(arrs)->logits(B,T,C), model_type, tokenizer, resize_mode).

    resize_mode は checkpoint の config.data.resize_mode から取得し、評価の前処理を訓練と
    一致させる(.onnx は config が無いので override か 'stretch')。
    """
    if model_path.suffix == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        mt, tok = _detect_model_type_and_tokenizer(sess)
        inp, outp = sess.get_inputs()[0].name, sess.get_outputs()[0].name

        def predict(arrs):
            x = np.stack(arrs)[:, None, :, :].astype(np.float32)
            return sess.run([outp], {inp: x})[0]
        return predict, mt, tok, (resize_mode_override or "stretch")
    else:
        import torch
        from meiban_ocr_trainer.models.crnn_pretrained import CRNNPretrained
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        full_cfg = ckpt.get("config", {})
        cfg = full_cfg.get("model", {})
        resize_mode = resize_mode_override or full_cfg.get("data", {}).get("resize_mode", "stretch")
        if cfg.get("arch") == "crnn_attn":
            from meiban_ocr_trainer.models.crnn_attn import CRNNAttn
            from meiban_ocr_trainer.tokenizer import AttnTokenizer
            model = CRNNAttn(hidden_size=int(cfg.get("crnn_hidden_size", 256)))
            model.load_state_dict(ckpt["model_state"])
            model = model.to(dev).eval()

            @torch.no_grad()
            def predict(arrs):
                x = torch.from_numpy(np.stack(arrs)[:, None, :, :].astype(np.float32)).to(dev)
                _, attn_logits = model(x, teacher_inputs=None)
                return attn_logits.cpu().numpy()
            return predict, "attn", AttnTokenizer(), resize_mode
        model = CRNNPretrained(
            num_classes=int(cfg.get("num_classes", 37)),
            hidden_size=int(cfg.get("crnn_hidden_size", 256)),
        )
        model.load_state_dict(ckpt["model_state"])
        model = model.to(dev).eval()

        @torch.no_grad()
        def predict(arrs):
            x = torch.from_numpy(np.stack(arrs)[:, None, :, :].astype(np.float32)).to(dev)
            return model(x).cpu().numpy()
        return predict, "ctc", CTCTokenizer(), resize_mode


def decode_batch(predict, mt, tok, arrs):
    logits = predict(arrs)
    return _decode_logits(mt, tok, logits)  # list[(text, conf)]


# ----- data -----
def read_rows(labels_path: Path, split: str):
    pos, neg = [], []
    with labels_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] != split:
                continue
            is_pos = (r.get("category") or "positive") == "positive" and (r.get("text") or "").strip()
            (pos if is_pos else neg).append(r)
    return pos, neg


def _rotate(bgr, deg):
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ----- evaluation -----
def evaluate(model_path: Path, root: Path, labels_path: Path, split: str,
             rotations, neg_conf: float, batch: int = 256,
             resize_mode_override: str | None = None) -> dict:
    from collections import Counter
    predict, mt, tok, resize_mode = load_predictor(model_path, resize_mode_override)
    pos_rows, neg_rows = read_rows(labels_path, split)

    # positives: 推論
    em = tot = 0
    cer_num = cer_den = 0
    errmult = Counter(); pos_err = Counter(); conf_pairs = Counter()
    n56 = 0; cc, cw = [], []
    by_ar = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}  # ar→ok
    q1_files = []  # 回転テスト用 (ar>=4.5)
    buf, meta = [], []

    def flush():
        nonlocal em, tot, cer_num, cer_den, n56
        if not buf:
            return
        preds = decode_batch(predict, mt, tok, buf)
        for (text, conf), (gt, ar) in zip(preds, meta):
            p = _norm(text); tot += 1
            d = _lev(p, gt); cer_num += d; cer_den += len(gt)
            ok = p == gt
            qkey = "Q1" if ar >= 4.5 else ("Q2" if ar >= 3.5 else ("Q3" if ar >= 2.3 else "Q4"))
            by_ar[qkey].append(ok)
            if ok:
                em += 1; cc.append(conf)
            else:
                cw.append(conf); errmult[min(d, 4)] += 1
                if len(p) == len(gt):
                    for i, (a, b) in enumerate(zip(gt, p)):
                        if a != b:
                            pos_err[i] += 1; conf_pairs[f"{a}->{b}"] += 1
                            if a == "5" and b == "6":
                                n56 += 1
        buf.clear(); meta.clear()

    for r in pos_rows:
        img = cv2.imread(str(root / r["filename"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        ar = w / h
        if ar >= 4.5:
            q1_files.append((r["filename"], _norm(r["text"])))
        buf.append(crop_and_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [0, 0, w, h], resize_mode))
        meta.append((_norm(r["text"]), ar))
        if len(buf) >= batch:
            flush()
    flush()

    # 回転耐性(Q1 サンプル)
    rng = random.Random(0)
    q1s = rng.sample(q1_files, min(400, len(q1_files))) if q1_files else []
    rot_em = {}
    for deg in rotations:
        ok = n = 0
        rb, rg = [], []
        for fn, gt in q1s:
            img = cv2.imread(str(root / fn))
            if img is None:
                continue
            b = _rotate(img, deg) if deg else img
            rb.append(crop_and_normalize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), [0, 0, b.shape[1], b.shape[0]], resize_mode))
            rg.append(gt)
            if len(rb) >= batch:
                for (t, _), g in zip(decode_batch(predict, mt, tok, rb), rg):
                    ok += _norm(t) == g; n += 1
                rb, rg = [], []
        if rb:
            for (t, _), g in zip(decode_batch(predict, mt, tok, rb), rg):
                ok += _norm(t) == g; n += 1
        rot_em[deg] = ok / n * 100 if n else 0.0

    # negatives: 誤発火
    nfire = nseen = 0
    nb = []
    for r in neg_rows:
        img = cv2.imread(str(root / r["filename"]))
        if img is None:
            continue
        nb.append(crop_and_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [0, 0, img.shape[1], img.shape[0]], resize_mode))
        if len(nb) >= batch:
            for t, c in decode_batch(predict, mt, tok, nb):
                nseen += 1
                if STRICT.match(_norm(t)) and c >= neg_conf:
                    nfire += 1
            nb = []
    if nb:
        for t, c in decode_batch(predict, mt, tok, nb):
            nseen += 1
            if STRICT.match(_norm(t)) and c >= neg_conf:
                nfire += 1

    return {
        "model": str(model_path), "labels": str(labels_path), "split": split,
        "resize_mode": resize_mode,
        "n_pos": tot, "n_neg": nseen,
        "EM": round(em / tot * 100, 2) if tot else 0,
        "CER": round(cer_num / cer_den * 100, 3) if cer_den else 0,
        "conf_correct_median": round(st.median(cc), 3) if cc else 0,
        "conf_wrong_median": round(st.median(cw), 3) if cw else 0,
        "err_multiplicity": dict(sorted(errmult.items())),
        "pos_errors": dict(sorted(pos_err.items())),
        "confusion_top": conf_pairs.most_common(10),
        "n_5to6": n56,
        "aspect_EM": {k: round(sum(v) / len(v) * 100, 1) for k, v in by_ar.items() if v},
        "rotation_EM": {f"{d}deg": round(rot_em[d], 1) for d in rotations},
        "neg_fire_rate": round(nfire / nseen * 100, 3) if nseen else 0,
    }


def _print(res: dict):
    print(f"\n=== {Path(res['model']).parent.name}/{Path(res['model']).name}  "
          f"[{Path(res['labels']).name}:{res['split']}] ===")
    print(f"  n_pos={res['n_pos']} n_neg={res['n_neg']} resize_mode={res.get('resize_mode')}")
    print(f"  EM={res['EM']}%  CER={res['CER']}%  5→6={res['n_5to6']}  neg誤発火={res['neg_fire_rate']}%")
    print(f"  conf median 正解={res['conf_correct_median']} 誤={res['conf_wrong_median']}")
    print(f"  アスペクト別EM={res['aspect_EM']}")
    print(f"  回転耐性={res['rotation_EM']}")
    print(f"  位置別誤り={res['pos_errors']}")
    print(f"  混同top={res['confusion_top']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="認識モデルの統合評価(serial-disjoint held-out)")
    p.add_argument("--model", type=Path, help="単一モデル(.pt または .onnx)")
    p.add_argument("--models", type=Path, nargs="+", help="複数モデル横並び")
    p.add_argument("--root", type=Path, default=Path("data/recognition"))
    p.add_argument("--labels", type=Path, default=Path("data/recognition/labels_serial_split.tsv"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--rotations", type=str, default="0,8,15")
    p.add_argument("--neg-conf-threshold", type=float, default=0.5)
    p.add_argument("--resize-mode", type=str, default=None, choices=("stretch", "letterbox"),
                   help="前処理リサイズを上書き(既定は checkpoint config の data.resize_mode)")
    p.add_argument("--compare-leaky", action="store_true",
                   help="リーク版 labels.tsv の同 split とも並記")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    models = args.models or ([args.model] if args.model else [])
    if not models:
        print("ERROR: --model か --models を指定", flush=True); return 1
    rotations = [int(x) for x in args.rotations.split(",")]

    results = []
    for m in models:
        r = evaluate(m, args.root, args.labels, args.split, rotations, args.neg_conf_threshold,
                     resize_mode_override=args.resize_mode)
        _print(r); results.append(r)
        if args.compare_leaky:
            leaky = args.root / "labels.tsv"
            if leaky.exists():
                rl = evaluate(m, args.root, leaky, args.split, rotations, args.neg_conf_threshold,
                              resize_mode_override=args.resize_mode)
                rl["note"] = "LEAKY(参考)"
                _print(rl); results.append(rl)
                print(f"  >>> clean EM={r['EM']}% vs leaky EM={rl['EM']}%  差={r['EM']-rl['EM']:+.1f}")

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
