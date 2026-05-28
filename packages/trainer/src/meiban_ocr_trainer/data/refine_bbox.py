"""VLM ラベリングの bbox を OpenCV で精密化する。Day 3 で実装。

LABELING.md トラブルシューティングを参照: VLM の bbox 精度は ±10〜30px なので、
ここで二値化 + 輪郭抽出で 1〜3px 精度に refine する。
"""

from __future__ import annotations


def refine_bbox(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("Day 3 で実装予定。")
