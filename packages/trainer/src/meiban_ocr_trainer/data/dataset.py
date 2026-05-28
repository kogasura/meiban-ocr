"""PyTorch Dataset / DataLoader for recognition crops.

`data/recognition/labels.tsv` を読み、各行を 1サンプルとして扱う。

labels.tsv columns (v2 schema):
- filename:   train/real/img_001_l00.png のような相対パス
- text:       ground truth 文字列 (negative は空文字)
- split:      train/val/test
- source:     real / replaced / synthetic / 画像stem
- confidence: 教師ラベル信頼度
- category:   positive | negative  (旧 v1 tsv で欠落していたら positive とみなす)
- subkind:    negative のみ (background | other_text | partial | other_vendor | mined)

訓練/評価で別 transform を使うため、split を引数で指定する。
CTC collate (可変長 target 連結 + category passthrough) は collate_fn として提供。

CTC は空 target (negative) を `zero_infinity=True` で正常に扱える。混在バッチでも
loss は有限値になることを test_dataset.py で検証している。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import cv2
import torch
from torch.utils.data import Dataset

from meiban_ocr_trainer.data.augment import build_eval_transform, build_train_transform, to_model_tensor
from meiban_ocr_trainer.tokenizer import CTCTokenizer


class RecognitionDataset(Dataset):
    """labels.tsv ベースの認識データセット (v2 schema 対応)。"""

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Return (image_tensor, sample_meta).

        sample_meta keys:
            text, category ('positive'|'negative'), subkind, source
        """
        row = self.rows[idx]
        img_path = self.root / row["filename"]
        arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        tensor = to_model_tensor(arr)
        meta = {
            "text": row.get("text") or "",  # negative は ""
            "category": row.get("category") or "positive",  # 旧 tsv は positive とみなす
            "subkind": row.get("subkind") or "",
            "source": row.get("source") or "",
        }
        return tensor, meta


def ctc_collate(
    batch: list[tuple[torch.Tensor, dict]],
    tokenizer: CTCTokenizer,
) -> dict:
    """CTC 訓練用 collate。category/subkind は per-sample 評価のため passthrough。

    Returns dict:
        images:         (B, 1, H, W)
        targets:        (sum(target_lengths),) long
        target_lengths: (B,) long (negative は 0)
        texts:          list[str] (positive は GT serial、negative は "")
        categories:     list[str] ('positive' | 'negative')
        subkinds:       list[str]
        sources:        list[str]
    """
    imgs = torch.stack([b[0] for b in batch])
    metas = [b[1] for b in batch]
    texts = [m["text"] for m in metas]
    targets, target_lengths = tokenizer.encode_batch(texts)
    return {
        "images": imgs,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
        "categories": [m["category"] for m in metas],
        "subkinds": [m["subkind"] for m in metas],
        "sources": [m["source"] for m in metas],
    }


def build_dataloaders(
    root: Path,
    tokenizer: CTCTokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, torch.utils.data.DataLoader]:
    """train/val/test の DataLoader を一括構築。

    train loader は通常の shuffle=True。curriculum (neg_ratio 動的制御) を使う場合は
    train_loop 側で本関数の train loader を捨て、`build_train_loader_with_ratio()` で
    epoch ごとに再構築する。
    """
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


def build_train_loader_with_ratio(
    train_ds: RecognitionDataset,
    tokenizer: CTCTokenizer,
    batch_size: int,
    neg_ratio: float,
    num_workers: int = 0,
    num_samples: int | None = None,
) -> torch.utils.data.DataLoader:
    """positive/negative 比率を `neg_ratio` に制御する WeightedRandomSampler 付き train loader。

    Args:
        train_ds: 訓練データセット
        batch_size: バッチサイズ
        neg_ratio: 1 batch あたりの期待 negative 割合 (0.0〜1.0)。0.0 だと positive only。
        num_workers: worker 数 (per-epoch 再構築されるので persistent_workers は無効)
        num_samples: epoch あたりサンプル数。None なら len(train_ds) × (1 batch 内
                     positive 数を確保するスケール) で算出。

    Why WeightedRandomSampler:
        small dataset で positive と negative の数に大きな偏りがある時、ratio を強制する
        手段。`replacement=True` で同じサンプルが複数回出ても OK (epoch 単位ではなく
        「ステップ数 = num_samples / batch_size」と解釈)。
    """
    from functools import partial

    from torch.utils.data import DataLoader, WeightedRandomSampler

    # 各サンプルの category
    cats = [row.get("category") or "positive" for row in train_ds.rows]
    n_pos = sum(1 for c in cats if c == "positive")
    n_neg = len(cats) - n_pos

    if n_neg == 0 or n_pos == 0:
        # 片方しかない → uniform sampling
        weights = [1.0] * len(cats)
    elif neg_ratio <= 0.0:
        # warmup 期: positive のみサンプリング (negative の weight を 0 にする)
        weights = [1.0 if c == "positive" else 0.0 for c in cats]
    elif neg_ratio >= 1.0:
        # 全 negative (理論上のみ、安全側に倒す)
        weights = [0.0 if c == "positive" else 1.0 for c in cats]
    else:
        # weight 設計: sum(weights for positive) = (1 - neg_ratio), sum(neg) = neg_ratio
        # → 各 positive の weight = (1 - neg_ratio) / n_pos, 各 negative = neg_ratio / n_neg
        pos_w = (1.0 - neg_ratio) / n_pos
        neg_w = neg_ratio / n_neg
        weights = [pos_w if c == "positive" else neg_w for c in cats]

    # epoch あたりステップ数: 元データセットと同じくらいの「1 epoch 感」を保つ
    if num_samples is None:
        num_samples = len(train_ds)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )

    collate = partial(ctc_collate, tokenizer=tokenizer)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )


def neg_ratio_for_epoch(schedule: list[dict], epoch: int) -> float:
    """curriculum schedule から指定 epoch の neg_ratio を線形補間で取得。

    schedule 例:
        [{"epoch": 1, "ratio": 0.0},
         {"epoch": 6, "ratio": 0.0},     # epoch 1-5 は完全 positive
         {"epoch": 16, "ratio": 0.30},   # epoch 6-15 で 0 → 0.30 線形上昇
         {"epoch": 40, "ratio": 0.40}]   # epoch 16-40 で 0.30 → 0.40 緩やかに

    epoch が schedule の範囲外 (前後) なら端点の値を返す。
    schedule が空なら 0.0 を返す。

    入力検証:
        - schedule 長 > 1000 で ValueError (config 経由 DoS 防止)
        - 各 ratio が [0, 1] 範囲外なら ValueError
    """
    if not schedule:
        return 0.0
    if len(schedule) > 1000:
        raise ValueError(
            f"neg_ratio_schedule too long ({len(schedule)} entries, max 1000)"
        )
    for s in schedule:
        ratio = float(s["ratio"])
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"neg_ratio_schedule entry has ratio={ratio} out of [0, 1]: {s}"
            )
    sorted_sched = sorted(schedule, key=lambda x: int(x["epoch"]))
    if epoch <= int(sorted_sched[0]["epoch"]):
        return float(sorted_sched[0]["ratio"])
    if epoch >= int(sorted_sched[-1]["epoch"]):
        return float(sorted_sched[-1]["ratio"])
    for i in range(len(sorted_sched) - 1):
        a, b = sorted_sched[i], sorted_sched[i + 1]
        ae, be = int(a["epoch"]), int(b["epoch"])
        if ae <= epoch <= be:
            if be == ae:
                return float(a["ratio"])
            t = (epoch - ae) / (be - ae)
            return float(a["ratio"]) + t * (float(b["ratio"]) - float(a["ratio"]))
    return float(sorted_sched[-1]["ratio"])
