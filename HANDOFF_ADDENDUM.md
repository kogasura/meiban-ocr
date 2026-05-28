# HANDOFF.md 追補 #1 — ベースモデル戦略の明確化

> このファイルは `HANDOFF.md` の追補です。既存ドキュメントを置き換えるものではなく、
> 重要な追加情報と1点のバグ修正を含みます。Claude Code は `HANDOFF.md` を読んだ後、
> このファイルも読んで反映してください。

---

## 変更点サマリ

1. **【バグ修正】** バックボーンの ImageNet 事前学習を有効化
2. **【追加】** ベースモデルの戦略を明文化 (現状は "Transfer Learning Lite")
3. **【追加】** Phase 1 で精度不足だった場合の Plan B/C/D を定義

---

## 1. バグ修正: ImageNet 事前学習を使う

`HANDOFF.md` の Section 3 のモデルコード:

```python
# ❌ 修正前
backbone = mobilenet_v3_small(weights=None)
```

これは **修正してください**:

```python
# ✅ 修正後
from torchvision.models import MobileNet_V3_Small_Weights
backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
```

### 理由

`weights=None` だとバックボーンも完全にゼロから訓練することになり、
- 訓練時間が大幅に長くなる
- 限られたデータで収束しにくくなる
- 汎化性能が落ちる

ImageNet 事前学習のバックボーンを使うことで:
- エッジ・テクスチャ・形状の汎用視覚特徴を継承
- OCR特化部分 (RNN + Classifier) だけを新規学習
- 訓練が速く・安定する

これは Transfer Learning の標準的な手法で、修正必須です。

---

## 2. ベースモデル戦略の明文化

現状の設計を整理すると **「部分的な事前学習 (Transfer Learning Lite)」** という構成です:

```
[MobileNetV3-Small (ImageNet pretrained) ← 事前学習済みで継承]
            ↓
[Bi-GRU 2層 ← ゼロから訓練]
            ↓
[Linear(256 → 37) ← ゼロから訓練]
            ↓
        CTC Loss
```

### なぜこの構成か

- **既存OCRモデル (PARSeq, TrOCR等) を fine-tune しない理由**:
  - 10MB〜100MBクラスのモデルが多く、サイズ目標 (<2MB) と合わない
  - アーキテクチャがViTベース等で、CRNN への流用が困難
  - 私たちはテキスト書き換えで10,000+サンプル確保できるので、ゼロから訓練可能

- **CNN バックボーンだけ事前学習**:
  - サイズは現状維持
  - 汎用特徴の継承で訓練の効率化
  - 業界標準のアプローチ

### Claude Code への補足指示

訓練スクリプト (`train.py`) では:
- バックボーンの学習率を低めに (1e-4)、RNN + Classifier の学習率を高めに (1e-3) する
  → バックボーンは既に良い特徴抽出器なので壊さない、頭部だけ強く学習させる
- 最初の数エポックはバックボーンを freeze し、頭部のみ訓練するウォームアップも有効

```python
# 学習率の差分設定例
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.rnn.parameters(),       'lr': 1e-3},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
])
```

---

## 3. Phase 1 で精度不足だった場合の代替戦略

Phase 1 の DoD (CER < 2%) を達成できなかった場合の判断ツリー:

```
評価結果を分析:
├─ 特定の文字パターンだけ苦手
│     → データ追加 (Phase 1.5、失敗ケース狙い撃ち)
│
├─ 全体的に精度が出ない (CER > 3%)
│     → Plan B: PARSeq fine-tune に切り替え検討
│
├─ 一部の難しいケースだけ間違える
│     → Plan D: 蒸留を Phase 4 で実施
│
└─ アーキテクチャの限界を疑う
      → Plan C: clovaai CRNN pretrained を試す
```

### Plan B: PARSeq fine-tune (精度最優先のとき)

```
ソース: NDLOCR-Lite の PARSeq モデル
URL: https://github.com/ndl-lab/ndlocr-lite
ファイル: parseq-ndl-16x256-30 (英文対応の実験版)
ライセンス: CC BY 4.0 (商用OK、帰属表示必須)
```

トレードオフ:
- メリット: 高精度、OCR知識を持つ、訓練が速い
- デメリット: モデルサイズが 10MB 程度 (INT8で 5MB 前後)、私たちの<2MB目標を超える
- **判断**: サイズ要件を 5MB に緩める許可をユーザーに取ってから実施

実装メモ:
- PARSeq の文字セットを A-Z, 0-9 (36文字) に絞る
- 入力サイズ 16×256 → 32×128 に合わせるか、入力側を調整
- バックボーンの学習率を下げて fine-tune

### Plan C: clovaai CRNN pretrained (アーキ近縁の流用)

```
URL: https://github.com/clovaai/deep-text-recognition-benchmark
モデル: None-ResNet-BiLSTM-CTC (CRNN系)
ライセンス: Apache 2.0
事前訓練データ: MJSynth + SynthText (英数字シーンテキスト)
```

トレードオフ:
- メリット: アーキテクチャが CRNN で近い
- デメリット: バックボーンが ResNet で大きい、MobileNetV3 への直接流用は不可
- **判断**: 同じ CRNN 設計のまま大きくする選択肢。サイズ目標との両立が難しいので Plan B の代替案

### Plan D: 知識蒸留 (最大精度を狙うとき、Phase 4)

```python
# 教師: PaddleOCR PP-OCRv5 mobile 認識モデル (ONNX で動かす)
# 生徒: 私たちの MobileNetV3 CRNN (現行のまま)

# 蒸留損失
distillation_loss = (
    alpha * KL_div(student_logits, teacher_logits / T) * T**2
    + (1 - alpha) * CE(student_logits, ground_truth)
)
# alpha = 0.7, T = 4.0 あたりが標準
```

トレードオフ:
- メリット: モデルサイズ据え置きで精度向上、教師の知識を凝縮
- デメリット: 実装複雑、教師の推論コスト、全データを教師に通す必要
- **判断**: Phase 4 で時間があれば実施。Phase 1〜3 のスコープ外

---

## 4. ドキュメントへの反映方針

Claude Code は以下を実施してください:

1. **`packages/trainer/src/meiban_ocr_trainer/models/tiny_ocr.py` の実装時**:
   - 上記の「修正後」コードを使う (ImageNet pretrained を有効化)
   - `train.py` で学習率の差分設定を入れる

2. **`HANDOFF.md` の更新は不要**:
   - この `HANDOFF_ADDENDUM.md` ファイルを残し、両方を参照する形で運用
   - 次のドキュメント全体更新時にマージする (今は触らない)

3. **Phase 1 DoD 未達時の対応**:
   - 上記の判断ツリーに従って Plan B/C/D を検討
   - 必ずユーザーに相談してから方針変更すること

---

## 5. その他の補足

### なぜ Plan B〜D を最初から採用しないか

- **Plan A (現状) で大半のケースは要件満足できる想定**
- Plan B/C/D は複雑さが増す = 開発・デバッグコストが増える
- まずシンプルな構成で動かして、不足なら段階的に複雑化する原則 (YAGNI)

### サイズ目標を緩める判断について

もしユーザーから「精度を最優先したい、サイズは5MB許容」と言われた場合は、
最初から Plan B (PARSeq fine-tune) を採用するのが妥当です。
その場合は HANDOFF.md の数値目標も更新が必要です (要相談)。
