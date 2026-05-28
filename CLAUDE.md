# CLAUDE.md

Claude Code がプロジェクト開始時に自動的に読むメモリファイル。
詳細仕様は `HANDOFF.md` を参照。

## プロジェクト

ブラウザで動く軽量な英数字OCRライブラリ。金属銘板の `E300MM000032` 形式の
シリアルコードを、カメラ1フレームから複数同時に抽出する。

- 訓練: PyTorch (CRNN: MobileNetV3-Small + Bi-GRU + CTC)
- 推論: onnxruntime-web (WebGPU + WASM)
- 公開: npm + HuggingFace Hub (Apache 2.0)

## アーキテクチャ

```
カメラフレーム → 検出(OpenCV.js) → 各ラベルをcrop → 認識(CRNN ONNX) → パターン補正
```

- **検出**: 古典CV (OpenCV.js) でラベル矩形抽出。MLは初期不要
- **認識**: 32×128 入力、CRNN、CTCデコード
- **文字セット**: A-Z, 0-9 の36文字 + CTC blank (汎用性のため全体で学習)
- **デコード**: ベンダー別パターン制約 + 6段階補正パイプライン

## ベンダーパターン (主対象 Ericsson)

- 厳格 regex: `/^E[39]\d{2}MM\d{6}$/` (例: `E300MM000032`)
- 本番DB 300,374件で**100%このパターンに一致** (悉皆調査済み) → デコード時の制約として強制可
- 6段階補正パイプライン (backend `PlateSerialNumber.php` と互換):
  1. 厳格完全一致 → 2. 厳格+O→0 → 3-4. 寛容 (Ericsson以外) → 5. 厳格部分+O→0 → 6. 厳格部分
- 前処理: NFKC + uppercase, `-` 除去
- マルチベンダー対応の余地を残す設計にする

## モノレポ構成

```
packages/trainer/   # Python 訓練側
packages/runtime/   # TypeScript 推論側 (npm公開対象)
models/             # 訓練済み ONNX
```

## 訓練データ戦略 (重要)

**動画ベース + テキスト書き換え**で効率的にデータ生成:

```
動画撮影 → フレーム抽出 → Claude Code ラベリング → テキスト書き換え水増し → 訓練
```

- 実画像 (動画フレーム + 直接撮影): 数百枚規模
- テキスト書き換えで 10〜50倍に水増し
- ランタイム augmentation で更に多様化

### データフォーマット (2段階)

**Stage 1**: 元画像 + アノテーション (1画像1JSON、Git管理)
```json
{ "image": "...", "image_size": [W,H], "source_video": "...",
  "labels": [{ "bbox": [...], "text_bbox": [...], "text": "E300MM000032", ... }] }
```

**Stage 2**: 認識訓練用クロップ (`extract_crops.py` で自動生成)
- `data/recognition/{train,val,test}/*.png` + `labels.tsv`
- train は real/replaced/synthetic に分類

### テストセット隔離 (重要)

`samples_test/` に隔離。訓練データに一切混入させない (Claude Code にも見せない)。
テキスト書き換え版は test に絶対使わない。

### データ量の目安

- 動画 5本 + 写真 50枚 + 書き換え 5000枚 → CER 0.5〜1% (MVP)
- 動画 10本 + 写真 100枚 + 書き換え 10000枚 → CER < 0.5% (プロダクション)

撮影は **動画ベース** が圧倒的に効率的。同じシーンを別端末・別バッチで撮ると効果大。

## 開発フェーズ

| Phase | 期間 | ゴール |
|---|---|---|
| 1 | Week 1〜2 | Claude Code ラベリング + CRNN訓練、CER < 2% |
| 1.5 | 任意 | 失敗ケース追加収集、CER < 1% |
| 2 | Week 3 | ONNX化 + INT8量子化、< 2MB |
| 3 | Week 4〜5 | TypeScript ランタイム、< 100ms/フレーム |
| 4 | Week 6〜7 | Fine-tune + 公開、完全一致率 > 98% |

各 Phase の DoD 達成前に次に進まない。

## 主要KPI

| 指標 | 目標 |
|---|---|
| 1フレーム全体 (検出+認識) | < 100ms (WebGPU) |
| モデル合計サイズ | < 5MB |
| 実画像 CER | < 0.5% |
| コード完全一致率 | > 98% |

## ルール

- 過剰な抽象化はしない (YAGNI)
- 多言語対応は Phase 4 まで先取りしない
- GPU前提の実装はしない (CPUでも訓練可能に)
- GPL系コードのコピー禁止 (Apache 2.0 / MIT / BSD のみ参考可)
- 依存追加は要相談 (`onnxruntime-web`, PyTorch標準, Albumentations, OpenCV, TRDG 以外)
- 公開APIの破壊的変更は要相談

## 報告タイミング

- 各 Phase の DoD 達成時
- 設計判断が必要な時
- 2時間以上詰まった時
- 想定外の発見があった時

## 参考実装 (ライセンス確認必須)

- `clovaai/deep-text-recognition-benchmark` (CRNN基準)
- `baudm/parseq` (モダンな認識モデル)
- `ndl-lab/ndlocr-lite` (ブラウザ実装の参考)

## ユーザー既存環境

React + Tesseract.js のカメラOCRアプリが稼働中。OpenCV.js 前処理、品質ゲート、
パターン補正は実装済み。`MeibanOCR` は**既存パイプラインに置換可能な形**で設計する。

## 言語

- ドキュメント・コメント・コミット: 日本語OK
- 公開API・README・関数名: 英語
