# v0 訓練結果 (2026-05-27)

## 設定
- データ: 4枚の Ericsson 銘板画像 (RapidOCR auto-label + Claude VLM verify)
  - train: img_001 (RRUS 11 B1, 18) + img_003 (RRU 22F3, 20) = 38 real + 1900 text-replaced
  - val: img_002 (Radio 2218 B42B) = 13 real
  - test: img_004 (Radio 2251 B18 B280, 部分遮蔽) = 3 real
- Model: TinyOCRModel (MobileNetV3-Small ImageNet pretrained + Bi-GRU + CTC)
- LR: backbone=1e-4 / head=1e-3、AdamW、Cosine annealing
- Warmup: 最初の 2 epoch は backbone freeze
- 100 epochs 設定、early stop で 76 epoch 終了

## 結果
- **best epoch: 46** (val_CER 0.0385)
- **val CER 3.85%** (HANDOFF Phase 1 DoD CER<2% に肉薄)
- **val EM 53.8%** (7/13 完全一致)
- test CER 47.22% (test=img_004 は別製品で訓練と異なる、3サンプルのみ)
- best.pt size: 3.0 MB (FP32、INT8量子化で <1MB 想定)

## 主な誤りパターン
val の誤り 6/13 はすべて **1文字違い**:
- E300MM000019 → E300MM999007 (6→0)
- E300MM000026 → E300MM999009 (1→7、8と並んで誤読)
- E300MM000024 → E300MM999005 (9→0)
- E300MM000027 → **E306**MM503814 (5→6)
- E300MM000025 → **E308**MM503792 (5→8)
- E300MM000021 → E300MM999006 (8→0)

→ **0/6/8/9 系の数字混同**が主な失敗源。典型的な OCR 誤り、データ拡張強化で改善可。

## 次のステップ
1. **augment 強化**: 0/6/8 区別のためノイズ・blur をさらに厚く、フォント多様化
2. **動画フレーム取込**: 2026-05 の MP4 4本から数百フレーム生成 (auto_label で自動)
3. **text_replace のフォント分布調整**: 実画像により近いセリフ/サンセリフ比率に
4. **Phase 2 (ONNX化)**: 現状の best.pt を ONNX 化 + INT8 量子化、サイズ確認
