"""RapidOCR を使った自動ラベリング (+ ハイブリッド・ダブルチェック対応)。

LABELING.md (Claude Code 用 VLM 手作業ラベリング) の代替。
URANUS2 OCR エンジン比較レポート (2026-05-27) で RapidOCR が対象画像に対し
100% (20/20) のカバレッジを達成したため、Phase 1 のラベル生成を RapidOCR に置き換える
(HANDOFF_ADDENDUM.md Plan D「蒸留」を Phase 1 から先取り)。

出力は annotation v2 schema (`regions[]`) に統一。RapidOCR は positive (Ericsson serial)
しか検出しないため、negative region は別途 (手作業 or hard negative mining) で追加する。

ダブルチェック方式 (--mode):
- `single`: RapidOCR 単独。後段で Claude VLM が目視検証する v0 用パス。
- `consensus`: RapidOCR + 第二エンジンの一致のみ採用 (scale 時の自動化パス)。
  第二エンジン (例: NDLOCR-Lite, PaddleOCR) の組み込みは TODO。
  両方が **同じ text** を **bbox IoU >= IOU_THRESHOLD** で検出した場合だけ採用、
  不一致は `disagreements` セクションに残して人間レビューに回す。

Usage:
    python -m meiban_ocr_trainer.data.auto_label \\
        --samples-dir samples/ \\
        --output-dir annotations/ \\
        --mode single \\
        [--vendor ericsson]

Notes:
- RapidOCR は polygon ベースで text 領域を返す。axis-aligned bbox に変換して保存。
- `bbox` (ラベル全体) は v0 では `text_bbox` をパディングしたもので近似 (refine_bbox.py で精密化予定)。
- regex フィルタで E[39]\\d{2}MM\\d{6} に一致するもののみ採用。一致しないテキストは捨てる。
- `claude_verified` フィールドは目視検証後に付与される (auto_label 自体は False のまま出力)。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable

from meiban_ocr_trainer.data.annotation import (
    Annotation,
    Region,
    load_annotation,
    save_annotation,
)

# Ericsson 厳格 regex (HANDOFF.md §2, vendorOcrPatterns.ts と一致)
ERICSSON_STRICT = re.compile(r"^E[39]\d{2}MM\d{6}$")
ERICSSON_PARTIAL = re.compile(r"E[39]\d{2}MM\d{6}")

# RapidOCR テキストの前処理: NFKC + uppercase + ハイフン除去 (PlateSerialNumber.php 互換)
import unicodedata


def _normalize(raw: str) -> str:
    return unicodedata.normalize("NFKC", raw).upper().replace("-", "").replace(" ", "")


def _poly_to_bbox(poly) -> tuple[int, int, int, int]:
    """RapidOCRの4点ポリゴンを axis-aligned bbox [x1,y1,x2,y2] に変換。"""
    xs = [int(round(p[0])) for p in poly]
    ys = [int(round(p[1])) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    pad_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    """text_bbox から bbox (ラベル全体) を概算するための padding。

    Why: ラベル銘板の境界はテキストより少し広い。pad_ratio は経験的に 15% で十分。
    refine_bbox.py (OpenCV) で精密化されるので、ここは概略でOK (LABELING.md トラブルシュート参照)。
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio * 2.5)  # 高さ方向は文字より広めに (label外枠想定)
    iw, ih = image_size
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(iw, x2 + pad_x),
        min(ih, y2 + pad_y),
    )


def _run_rapidocr(ocr, image_path: Path) -> tuple[list[tuple], float]:
    """RapidOCR を1画像にかけ、エントリと経過秒を返す。各エントリ: (poly, raw_text, conf)。"""
    t0 = time.time()
    result, _ = ocr(str(image_path))
    elapsed = time.time() - t0
    return (result or []), elapsed


def _extract_serial_candidates(
    raw_entries: Iterable[tuple],
) -> list[dict]:
    """OCR エントリを Ericsson regex でフィルタし、shape-normalized 辞書のリストを返す。

    各辞書: {text, conf, match_kind, text_bbox: (x1,y1,x2,y2)}
    """
    out: list[dict] = []
    for entry in raw_entries:
        poly, raw_text, conf = entry[0], entry[1], float(entry[2])
        text = _normalize(raw_text)
        if ERICSSON_STRICT.match(text):
            kind = "strict"
        else:
            m = ERICSSON_PARTIAL.search(text)
            if not m:
                continue
            text = m.group(0)
            kind = "partial"
        out.append({
            "text": text,
            "conf": conf,
            "match_kind": kind,
            "text_bbox": _poly_to_bbox(poly),
            "raw_text": raw_text,
        })
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union else 0.0


def auto_label_image(
    image_path: Path,
    primary_ocr,
    *,
    secondary_ocr=None,
    vendor: str = "ericsson",
    iou_threshold: float = 0.5,
) -> Annotation:
    """1画像をOCRに通し、Annotation を生成。secondary_ocr 指定時は consensus モード。

    RapidOCR は Ericsson regex に一致する文字列のみ検出するため、出力は全 positive。
    negative region は本関数では生成しない (別途手作業 or mining で追加)。
    """
    from PIL import Image

    im = Image.open(image_path)
    w, h = im.size

    raw_p, elapsed_p = _run_rapidocr(primary_ocr, image_path)
    cands_p = _extract_serial_candidates(raw_p)

    if secondary_ocr is None:
        # single モード: 全候補をそのまま採用
        accepted = cands_p
        disagreements: list[dict] = []
        elapsed_s = 0.0
        n_secondary = 0
    else:
        # consensus モード: 第二エンジンの結果と突き合わせ
        raw_s, elapsed_s = _run_rapidocr(secondary_ocr, image_path)
        cands_s = _extract_serial_candidates(raw_s)
        n_secondary = len(cands_s)
        accepted = []
        disagreements = []
        s_matched = [False] * len(cands_s)
        for cp in cands_p:
            match_idx = -1
            best_iou = 0.0
            for j, cs in enumerate(cands_s):
                if s_matched[j]:
                    continue
                if cp["text"] != cs["text"]:
                    continue
                iou = _iou(cp["text_bbox"], cs["text_bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    match_idx = j
            if match_idx >= 0:
                s_matched[match_idx] = True
                merged = dict(cp)
                merged["secondary_conf"] = cands_s[match_idx]["conf"]
                merged["iou"] = round(best_iou, 4)
                accepted.append(merged)
            else:
                disagreements.append({"engine": "primary_only", **cp})
        for j, cs in enumerate(cands_s):
            if not s_matched[j]:
                disagreements.append({"engine": "secondary_only", **cs})

    regions: list[Region] = []
    for c in accepted:
        text_bbox = c["text_bbox"]
        bbox = _pad_bbox(text_bbox, (w, h))
        # Why: is_clear == confidence>=0.8 を quality に対応付ける (v1 互換)。
        quality = "clear" if c["conf"] >= 0.8 else "blur"
        extra: dict = {}
        if "secondary_conf" in c:
            extra["secondary_confidence"] = round(c["secondary_conf"], 4)
            extra["iou_with_secondary"] = c["iou"]
        regions.append(Region(
            id=len(regions),
            category="positive",
            bbox=list(bbox),
            text_bbox=list(text_bbox),
            text=c["text"],
            vendor=vendor,
            quality=quality,
            confidence=round(c["conf"], 4),
            match_kind=c["match_kind"],
            claude_verified=False,
            extra=extra,
        ))

    n_rejected_p = len(raw_p) - len(cands_p)

    meta: dict = {
        "ocr_engine": "rapidocr_onnxruntime",
        "mode": "consensus" if secondary_ocr is not None else "single",
        "ocr_elapsed_sec": {"primary": round(elapsed_p, 3), "secondary": round(elapsed_s, 3)},
        "rejected_by_regex_primary": n_rejected_p,
    }
    if secondary_ocr is not None:
        meta["secondary_engine"] = "TBD"
        meta["secondary_candidates_count"] = n_secondary
        meta["disagreements"] = disagreements

    return Annotation(
        image=image_path.name,
        image_size=[w, h],
        source_video=None,
        vendor=vendor,
        regions=regions,
        meta=meta,
    )


def write_report(annotations: list[Annotation], output_dir: Path) -> None:
    """LABELING.md §Step 4 と同等の _report.md を生成。

    positive region のみ集計 (RapidOCR は positive のみ、negative は別経路)。
    """
    n_images = len(annotations)
    pos_per_image = [a.positives for a in annotations]
    n_pos = sum(len(p) for p in pos_per_image)
    if n_images == 0:
        return
    n_neg = sum(len(a.negatives) for a in annotations)

    pos_flat = [r for p in pos_per_image for r in p]
    high_conf = sum(1 for r in pos_flat if (r.confidence or 0.0) >= 0.95)
    mid_conf = sum(1 for r in pos_flat if 0.80 <= (r.confidence or 0.0) < 0.95)
    low_conf = sum(1 for r in pos_flat if (r.confidence or 0.0) < 0.80)
    clear_true = sum(1 for r in pos_flat if r.quality == "clear")
    clear_false = n_pos - clear_true

    all_serials = [r.text for r in pos_flat]
    duplicates = sorted({s for s in all_serials if all_serials.count(s) > 1})

    per_image = sorted(
        ((Path(a.image).name, len(a.positives)) for a in annotations),
        key=lambda x: -x[1],
    )

    total_elapsed = sum(
        a.meta.get("ocr_elapsed_sec", {}).get("primary", 0.0) for a in annotations
    )

    md = [
        "# Labeling Report (auto-generated by RapidOCR)",
        "",
        f"- 処理画像数: {n_images}",
        f"- 検出 positive 総数: {n_pos}",
        f"- 既存 negative 総数: {n_neg}",
        f"- 平均 positive 数/画像: {n_pos / n_images:.1f}",
        f"- 合計OCR時間 (primary): {total_elapsed:.2f}s",
        "",
        "## 信頼度分布 (positive)",
        f"- confidence >= 0.95: {high_conf}件 ({100*high_conf/max(n_pos,1):.0f}%)",
        f"- 0.80 <= confidence < 0.95: {mid_conf}件",
        f"- confidence < 0.80: {low_conf}件",
        "",
        "## quality 分布 (positive)",
        f"- clear: {clear_true}件",
        f"- blur:  {clear_false}件 (要確認)",
        "",
        "## 警告",
        f"- 重複しているシリアル: {duplicates or 'なし'}",
        "",
        "## サンプル別 positive 数",
    ]
    for name, n in per_image:
        md.append(f"- {name}: {n} labels")
    (output_dir / "_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-label nameplate images with RapidOCR")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--vendor", type=str, default="ericsson")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("single", "consensus"),
        default="single",
        help="single=RapidOCR単独 (v0でClaudeが後段検証), consensus=2エンジン一致 (scale用、TODO)",
    )
    parser.add_argument(
        "--exts", type=str, default="jpg,jpeg,png", help="comma-separated extensions"
    )
    args = parser.parse_args(argv)

    if not args.samples_dir.is_dir():
        print(f"[auto_label] samples dir not found: {args.samples_dir}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR

    print("[auto_label] loading RapidOCR…", file=sys.stderr)
    primary = RapidOCR()
    secondary = None
    if args.mode == "consensus":
        # Why: 第二エンジン (NDLOCR-Lite / PaddleOCR) の組み込みは scale フェーズの作業。
        # 現状は同じ RapidOCR を 2 回呼ぶ (sanity-check 用のダミー consensus)。
        # 本番では別エンジンに差し替える。
        print("[auto_label] WARNING: consensus mode uses RapidOCR twice (TODO: integrate NDLOCR-Lite)", file=sys.stderr)
        secondary = primary

    exts = {f".{e.lower()}" for e in args.exts.split(",")}
    images = sorted(p for p in args.samples_dir.iterdir() if p.suffix.lower() in exts)
    print(f"[auto_label] {len(images)} images to process (mode={args.mode})", file=sys.stderr)

    annotations: list[Annotation] = []
    for img in images:
        out_json = args.output_dir / f"{img.stem}.json"
        if out_json.exists():
            print(f"  - {img.name}: SKIP (already labeled)", file=sys.stderr)
            # 旧 v1 形式でも load_annotation が positive Region に変換する
            annotations.append(load_annotation(out_json))
            continue
        try:
            ann = auto_label_image(
                img, primary, secondary_ocr=secondary, vendor=args.vendor
            )
        except Exception as e:  # noqa: BLE001
            print(f"  - {img.name}: ERROR {e}", file=sys.stderr)
            continue
        annotations.append(ann)
        save_annotation(ann, out_json)
        elapsed = ann.meta.get("ocr_elapsed_sec", {}).get("primary", 0.0)
        print(
            f"  - {img.name}: {len(ann.positives)} labels, "
            f"primary {elapsed:.1f}s",
            file=sys.stderr,
        )

    write_report(annotations, args.output_dir)
    print(f"[auto_label] done. report: {args.output_dir / '_report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
