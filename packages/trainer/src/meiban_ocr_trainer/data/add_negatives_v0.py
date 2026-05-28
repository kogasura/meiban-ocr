"""v0 negative annotation 追加 (one-shot script for img_001..004)。

既存 4 画像に対し、目視で安全と確認した負例 bbox を追記する。
- val (img_002) / test (img_004) を優先 (reject 評価サンプル)
- train (img_001 / img_003) も最低限の hard negative を追加 (model に "出さない" を学ばせる)

各 bbox は positive と重ならない領域を選定済み (画像目視 + 既存 bbox との非交差を検証)。
本スクリプトは冪等: 既に同 id の negative がある場合はスキップする。

Usage:
    python -m meiban_ocr_trainer.data.add_negatives_v0
"""

from __future__ import annotations

import sys
from pathlib import Path

from meiban_ocr_trainer.data.annotation import (
    Annotation,
    Region,
    load_annotation,
    save_annotation,
)

# (image_stem, [(bbox, subkind, text_visible), ...])
NEGATIVES_TO_ADD: dict[str, list[tuple[list[int], str, str | None]]] = {
    "img_002": [
        # 画像 1280×960。13 positive (左6 + 中2 + 中2 + 右4)。
        # 周辺の dark background + 各ラベル内の non-serial 行を追加。

        # background: 画像四隅・カラム間の空き
        ([20, 20, 250, 110], "background", None),
        ([600, 30, 950, 130], "background", None),
        ([1180, 10, 1270, 200], "background", None),
        ([400, 720, 700, 820], "background", None),
        ([950, 720, 1250, 820], "background", None),
        ([20, 720, 320, 850], "background", None),
        # other_text: ラベル内の Radio 2218 B42B 行 (positive の上)
        # row0 left  (positive at [142,283,324,322]) → Radio 行は y ≈ 252-273, gap 上端 OK
        ([145, 252, 320, 273], "other_text", "Radio 2218 B42B"),
        ([427, 257, 600, 280], "other_text", "Radio 2218 B42B"),
        ([709, 248, 884, 270], "other_text", "Radio 2218 B42B"),
        ([1012, 232, 1184, 254], "other_text", "Radio 2218 B42B"),
        # other_text: エリクソン footer (各ラベルの下、次ラベルの上の gap 内)
        # row0 left positive ends y=322, row1 starts y=345 → footer 行は y=328-348 で安全
        ([145, 328, 320, 348], "other_text", "エリクソン・ジャパン株式会社"),
        ([427, 335, 600, 355], "other_text", "エリクソン・ジャパン株式会社"),
        # 製造番号/製造年月 ラベル (各ラベルの左側) ※ positive と完全に gap で離れている所のみ
        # row0 left  positive [142,283,324,322]
        # ところで「製造番号」は serial の左にあって positive bbox 内部にあるので
        # ここを取ると CTC が混乱する → 取らない。
        # 代わりに最下段 left positive (y=618-652) の下の footer 強行採用:
        ([145, 658, 322, 680], "other_text", "エリクソン・ジャパン株式会社"),
    ],
    "img_004": [
        # 画像 1200×1600。3 positive (全部 x=750-1170, 上中下)。
        # 左半分は白カード (手書き) + 右下は袋背景 → 多くの空き領域あり。

        # background: ウッド/デスク表面
        ([20, 1450, 350, 1580], "background", None),
        ([900, 1450, 1180, 1580], "background", None),
        # other_text: 白カード上の手書きテキスト
        # "503509" 手書き: 画像右側中段、横倒し気味
        ([350, 100, 700, 260], "other_text", "503509 (handwritten)"),
        # "503505" 手書き: 中央付近
        ([400, 420, 720, 600], "other_text", "503505 (handwritten)"),
        # NOTE: "E326MM503410" 印字 (縦書き) は Ericsson strict pattern と一致してしまい、
        # 負例として学習させると pattern 認識を破壊するので除外。手書きの個別数字 (503509,
        # 503505) は pattern 全体に一致しないため負例化 OK。
        # 白カードのピンクドット領域 (テクスチャ無しほぼ単色)
        ([130, 270, 230, 360], "background", None),
        # 袋背景 (positive ラベルの間、textがほぼ無い領域)
        # positive 0 [768,9,1138,103] と positive 1 [775,574,1161,698] の間 (y=104-573)
        # 中央付近に reflective な空き
        ([800, 180, 1100, 300], "background", None),
        ([800, 400, 1100, 540], "background", None),
    ],
    "img_001": [
        # 画像 1280×960。18 positive (8行 × 多列に近い grid)。
        # train 用なので少なめに、確実に安全な背景のみ。

        # background: 画像四隅
        ([20, 20, 200, 200], "background", None),
        ([1100, 20, 1270, 200], "background", None),
        ([20, 850, 200, 950], "background", None),
        ([800, 850, 1100, 950], "background", None),
        # 中央余白 (label と label の間の dark area)
        # 大半の positive は y=355-723 に集中、x=103-1203
        # 安全マージンとして y < 340 or y > 740 のみ採用
        ([400, 30, 700, 120], "background", None),
        ([700, 850, 1000, 950], "background", None),
    ],
    "img_003": [
        # 画像 1280×960。20 positive。
        # train 用、最低限の背景のみ。
        ([20, 20, 200, 120], "background", None),
        ([1100, 20, 1270, 200], "background", None),
        ([20, 850, 250, 950], "background", None),
        ([1050, 850, 1270, 950], "background", None),
    ],
}


def _bbox_overlaps(a: list[int], b: list[int]) -> bool:
    """[x1,y1,x2,y2] 同士の交差判定。"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _add_negatives_to(ann: Annotation, negs: list[tuple[list[int], str, str | None]]) -> int:
    """既存 positive と重ならない negative のみを Annotation に追加。

    Returns:
        実際に追加された件数。
    """
    pos_bboxes = [r.bbox for r in ann.positives]
    # 既存 negative の bbox 集合 (idempotency 用)
    existing_neg = {tuple(r.bbox) for r in ann.negatives}

    next_id = max((r.id for r in ann.regions), default=-1) + 1
    added = 0
    for bbox, subkind, text_visible in negs:
        # 冪等性: 同じ bbox の negative が既にあればスキップ
        if tuple(bbox) in existing_neg:
            continue
        # positive 衝突チェック
        conflict = next((p for p in pos_bboxes if _bbox_overlaps(bbox, p)), None)
        if conflict is not None:
            print(
                f"  ! SKIP bbox={bbox} (overlaps positive {conflict})",
                file=sys.stderr,
            )
            continue
        ann.regions.append(Region(
            id=next_id,
            category="negative",
            bbox=list(bbox),
            subkind=subkind,  # type: ignore[arg-type]
            text_visible=text_visible,
            claude_verified=True,
        ))
        next_id += 1
        added += 1
    return added


def main() -> int:
    annotations_dir = Path("annotations")
    if not annotations_dir.is_dir():
        print(f"[add_negatives] not found: {annotations_dir}", file=sys.stderr)
        return 1

    total = 0
    for stem, negs in NEGATIVES_TO_ADD.items():
        path = annotations_dir / f"{stem}.json"
        if not path.exists():
            print(f"  - {stem}: SKIP (no annotation file)", file=sys.stderr)
            continue
        ann = load_annotation(path)
        n_before = len(ann.negatives)
        added = _add_negatives_to(ann, negs)
        save_annotation(ann, path)
        n_after = len(ann.negatives)
        print(
            f"  - {stem}: {n_before} → {n_after} negatives (+{added})",
            file=sys.stderr,
        )
        total += added
    print(f"[add_negatives] total added: {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
