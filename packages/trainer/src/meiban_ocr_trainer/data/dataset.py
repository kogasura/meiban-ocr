"""PyTorch Dataset / DataLoader for recognition crops.

`data/recognition/labels.tsv` を読み、各行を 1サンプルとして扱う。
- filename: train/real/img_001_l00.png のような相対パス
- text: ground truth 文字列
- split: train/val/test
- source: real / replaced / synthetic / 画像stem
- confidence: 教師ラベル信頼度

訓練/評価で別 transform を使うため、split を引数で指定する。
CTC collate (可変長 target 連結) は collate_fn として提供。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from meiban_ocr_trainer.data.augment import build_eval_transform, build_train_transform, to_model_tensor
from meiban_ocr_trainer.tokenizer import CTCTokenizer


class RecognitionDataset(Dataset):
    """labels.tsv ベースの認識データセット。"""

    def __init__(
        self,
        root: Path,
        split: str,
        tokenizer: CTCTokenizer | None = None,
        transform: Callable | None = None,
        labels_filename: str = "labels.tsv",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.tokenizer = tokenizer or CTCTokenizer()
        self.transform = transform

        labels_path = self.root / labels_filename
        if not labels_path.exists():
            raise FileNotFoundError(f"labels file not found: {labels_path}")

        rows: list[dict] = []
        with labels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["split"] != split:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f"no rows for split={split} in {labels_path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.rows[idx]
        img_path = self.root / row["filename"]
        # cv2.imread は BGR
        arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        tensor = to_model_tensor(arr)
        return tensor, row["text"]


def ctc_collate(
    batch: list[tuple[torch.Tensor, str]],
    tokenizer: CTCTokenizer,
) -> dict[str, torch.Tensor | list[str]]:
    """CTC 訓練用 collate。

    Returns dict:
        images: (B, 1, H, W)
        targets: (sum(target_lengths),) long
        target_lengths: (B,) long
        texts: list[str] (ground truth 文字列、デバッグ用)
    """
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    targets, target_lengths = tokenizer.encode_batch(texts)
    return {
        "images": imgs,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
    }


def build_dataloaders(
    root: Path,
    tokenizer: CTCTokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, torch.utils.data.DataLoader]:
    """train/val/test の DataLoader を一括構築。"""
    from functools import partial

    from torch.utils.data import DataLoader

    train_ds = RecognitionDataset(root, "train", tokenizer, build_train_transform())
    val_ds = RecognitionDataset(root, "val", tokenizer, build_eval_transform())
    test_ds = RecognitionDataset(root, "test", tokenizer, build_eval_transform())

    collate = partial(ctc_collate, tokenizer=tokenizer)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            collate_fn=collate,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        ),
    }
