"""動画から疎にフレーム抽出 + ブラー判定で軽くフィルタ。Day 2 で実装。

HANDOFF.md §4 Step 1 を参照。雛形のみ。
"""

from __future__ import annotations

from pathlib import Path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps_sample: float = 1.0,
    sharpness_threshold: float = 100.0,
) -> int:
    """動画から疎にフレーム抽出。

    Args:
        video_path: 入力動画
        output_dir: フレーム書き出し先 (`samples/`)
        fps_sample: 抽出fps。1.0以下推奨 (連続フレームは超相関のため)
        sharpness_threshold: Laplacian variance 閾値。緩めに (ブラーも訓練に必要)

    Returns:
        書き出したフレーム数。
    """
    raise NotImplementedError("Day 2 で実装予定。HANDOFF.md §4 Step 1 を参照。")
