# Edge-Optimized OCR Library — 引き継ぎドキュメント

ブラウザで動く軽量な英数字OCRライブラリを自作するプロジェクトの仕様書。
Claude Code は最初にこれを通読してから作業を開始すること。

---

## 1. プロジェクト概要

**作るもの**: モバイルブラウザで動く、軽量・高速な英数字コードOCRライブラリ
**対象**: 金属銘板の `E300MM000032` のような英数字シリアルコード (画像参照)
**技術**: PyTorch訓練 → ONNX変換 → onnxruntime-web 実行
**最終形態**: npm パッケージ + HuggingFace Hub 上のモデル

### ユースケース

カメラを銘板シートにかざすと、視野内の複数ラベルから**3秒で10件以上**のコードを連続抽出。
シリアルコード収集の業務効率化が目的。

### 数値目標

| 指標 | 目標 |
|---|---|
| 検出 + 認識 合計時間 (1フレーム, ラベル多数) | < 100ms (WebGPU) |
| 1コードあたりの認識 | < 20ms |
| モデルサイズ合計 (検出 + 認識, INT8) | < 5MB |
| 実画像CER | < 0.5% |
| コード完全一致率 | > 98% |

### 既存環境

ユーザーは React + Tesseract.js のカメラOCRアプリを既に運用中。
固定スキャン枠、品質ゲート (Laplacian variance)、OpenCV.js 前処理、
パターン補正 (`symbols[].choices`) などは実装済み。
本ライブラリは**既存パイプラインに置換可能な形**で設計すること。

---

## 2. 認識対象の仕様

### 対象画像の特徴 (実サンプル)

- 銀色金属光沢の矩形ラベル (アスペクト比 約 2.5:1 〜 3:1)
- 1フレームに複数のラベル (グリッド配置のことが多い)
- 各ラベルに以下の情報:
  - 型番: `RRU 22F3` (固定文字列の可能性高)
  - 製造番号: `E300MM000032` 形式 (英数字12文字)
  - 製造年月: `2018年9月` 形式
  - 会社名: `エリクソン・ジャパン株式会社`

### 認識対象は「製造番号」のみ

MVP では **英数字12文字のシリアルコードのみ** を対象とする。
他の情報 (型番、日付、会社名) は将来要望次第。

### ベンダー別パターン (Ericsson が主対象)

**Ericsson (VENDOR_ID=2)**:
- 厳格 regex: `/^E[39]\d{2}MM\d{6}$/`
- 部分一致 regex: `/E[39]\d{2}MM\d{6}/`
- フォーマット: `E` + `3 or 9` + 数字2桁 + `MM` + 数字6桁
- 例: `E300MM999001`, `E300MM000032`
- 本番DB 300,374件の悉皆調査で**100%このパターンに一致** (検証済み)
- → デコード時の位置別文字種制約として強制してよい

**マルチベンダー設計**: 将来的に他ベンダー対応の想定 (現状は Ericsson のみ)。
パターン定義は外部から差し替え可能な形で保持する:

```typescript
interface VendorPattern {
  vendorId: number;
  vendorName: string;
  strictRegex: RegExp;
  partialRegex: RegExp;
  positionConstraints?: Array<Set<string>>; // 位置別許容文字
}
```

訓練時は **36文字全体** で学習 (汎用性確保)、デコード時にパターン制約を適用する設計とする。

---

## 3. アーキテクチャ

### 全体パイプライン

```
[カメラフレーム]
  ↓
[① 検出: 古典CV でラベル矩形を抽出]   ← OpenCV.js, 数ms
  ↓
[② 各矩形を 32×128 にリサイズ + 正規化]
  ↓
[③ 認識: CRNN モデル]                  ← ONNX, WebGPU
  ↓
[④ パターン検証 + 重複排除]
  ↓
[結果: 検出した複数コード]
```

### 検出 (Detection)

金属ラベルは形状が単純なので、**学習なしの古典CVで第一段**を組む。

```
1. cvtColor → グレースケール
2. adaptiveThreshold (or Otsu) で二値化
3. morphologyEx (closing) でラベル領域を埋める
4. findContours で輪郭抽出
5. boundingRect + アスペクト比・サイズフィルタ
6. 必要なら perspective transform で水平正規化
```

ラベルは背景とのコントラストが極端なので、これで 90%+ 検出できる想定。
失敗ケースが多いなら Phase 4 で軽量検出モデル (MobileNetV3 + bbox回帰) を追加検討。

**学習済み検出モデルは初期は作らない**。古典CVで足りるか実データで検証してから判断。

### 認識 (Recognition)

CRNN (MobileNetV3-Small + Bi-GRU + CTC):

```python
class TinyOCRModel(nn.Module):
    """
    入力: (B, 1, 32, 128) グレースケール、[0,1] 正規化
    出力: (B, T, 37) logits (36 chars + CTC blank)
    """
    def __init__(self, num_classes=37):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(*list(backbone.features.children())[:10])
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.GRU(
            input_size=96, hidden_size=128, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.1
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).squeeze(2).permute(0, 2, 1)
        out, _ = self.rnn(feat)
        return self.classifier(out)
```

### 文字セット

```python
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 36 chars
BLANK_IDX = 36
NUM_CLASSES = 37
```

TypeScript側でも同じ定義を共有 (`constants.ts` で管理)。

### CTC デコード + 6段階補正パイプライン

Phase 1〜2 は Greedy CTC で開始。Phase 3 のランタイムでは backend
(`server/app/Domain/Nameplate/PlateSerialNumber.php`) と互換の補正パイプラインを実装:

```
1. 厳格 完全一致
2. 厳格 完全一致 + O→0 フォールバック (OCRがO/0を誤読しやすいため)
3. (寛容 完全一致 — Ericsson以外のベンダー用)
4. (寛容 完全一致 + O→0 — Ericsson以外用)
5. 厳格 部分一致 + O→0
6. 厳格 部分一致
```

**前処理**: NFKC normalize + uppercase, `-` を除去
(例: `E325-MM-004005` → `E300MM999019` でマッチ)

backend が source of truth なので、ランタイム実装はその挙動を踏襲する。
TypeScript と Python の両方でテストケースを揃え、回帰検出できるようにする。

---

## 4. データ戦略

### 全体パイプライン

```
[動画撮影 5-10本]
     ↓ extract_frames.py
[フレーム抽出 500-1000枚] → samples/
     ↓ Claude Code (LABELING.md)
[アノテーション] → annotations/*.json
     ↓ extract_crops.py
[認識用クロップ] → data/recognition/
     ↓ text_replace.py
[テキスト書き換えで 10x-50x 水増し]
     ↓ ランタイム augmentation (Albumentations)
[訓練]
```

**実画像撮影は最小限**で済む。動画でラクに量を稼ぎ、テキスト書き換えで文字多様性を補う。

### Step 1: 動画撮影 + フレーム抽出

ユースケースは「真上付近・似た照明」固定なので、生の実画像をたくさん撮るより
**動画ベースで効率的に量を稼ぐ**:

```
動画10本 × 1分 × 1fps = 600フレーム
```

撮影レシピ:
- シーン1〜3: 平面ラベルを接写でスイープ (各1分)
- シーン4〜5: 引きで全体を映す (各1分)
- シーン6〜7: 別バッチのラベルで同様
- シーン8〜10: 違うスマホで再撮影、軽い角度違い等

```python
# packages/trainer/src/meiban_ocr_trainer/data/extract_frames.py
import cv2
from pathlib import Path

def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps_sample: float = 1.0,
    sharpness_threshold: float = 100.0
) -> int:
    """動画から疎にフレーム抽出 + ブラー判定で軽くフィルタ"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / fps_sample))
    
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness > sharpness_threshold:
                out = output_dir / f"{video_path.stem}_f{saved:05d}.jpg"
                cv2.imwrite(str(out), frame)
                saved += 1
        frame_idx += 1
    cap.release()
    return saved
```

注意点:
- `fps_sample` は 1.0 以下推奨 (連続フレームは超相関)
- ブラー判定は **緩めに** (ブラー画像も訓練データとして必要)
- **同じ動画から train/val 両方に振らない** (動画単位で分割すること)

### Step 2: Claude Code でラベリング

抽出フレームを `samples/` に集約し、Claude Code セッションで `LABELING.md` を実行。
詳細は別途 `LABELING.md` 参照。

### Step 3: テキスト書き換えで水増し (重要)

実画像で集まるシリアルは連番なので文字バランスに偏りが出る (例: `E303MM5xxxxx` ばかり)。
**ラベルクロップの中のシリアルを別文字列に置き換え**て、文字多様性を確保:

```python
# packages/trainer/src/meiban_ocr_trainer/data/text_replace.py
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

def text_replace(
    label_crop: np.ndarray,
    text_bbox: tuple,    # クロップ内のシリアル領域 (x1,y1,x2,y2)
    new_serial: str,
    font_path: str = 'fonts/OCR-B.ttf',
) -> np.ndarray:
    """ラベル画像のシリアル文字列を別の文字列に置換"""
    img = label_crop.copy()
    x1, y1, x2, y2 = text_bbox
    
    # 1. Inpainting で元テキストを背景化
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    # 2. 新テキストを描画
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_size = int((y2 - y1) * 0.7)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((x1, y1), new_serial, font=font, fill=(20, 20, 20))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    # 3. 軽いブラー + ノイズで馴染ませる
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def generate_random_ericsson_serial() -> str:
    """Ericsson パターンに従う架空シリアル生成"""
    import random
    prefix = random.choice(['E3', 'E9'])
    middle = ''.join(random.choices('0123456789', k=2))
    suffix = ''.join(random.choices('0123456789', k=6))
    return f"{prefix}{middle}MM{suffix}"
```

各実画像クロップから 10〜50バリエーション生成 → **実画像 1枚 = 数十枚相当の訓練データ**。

ポイント:
- **フォントは完璧一致を狙わない** (OCR-A / OCR-B / Consolas など産業印字風で OK)
- 文字バランスを統計的に均一化する (E3/E9 を 50:50、各数字を均等出現に)
- **test セットには使わない** (合成データなので実環境評価にならない)

### Step 4: ランタイム augmentation (Albumentations)

訓練時に動的に適用:

```python
import albumentations as A

train_transform = A.Compose([
    # 幾何 (真上付近なので控えめ)
    A.Affine(rotate=(-2, 2), scale=(0.92, 1.08), shear=(-2, 2), p=0.4),
    A.Perspective(scale=(0.01, 0.04), p=0.3),
    
    # 照明 (大差ないとはいえ多少のバリエーション)
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    
    # 質感劣化 (実機の画質を模擬、ここを厚くする)
    A.ImageCompression(quality_lower=25, quality_upper=85, p=0.6),
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.4),
    
    # 部分欠け (反射・汚れ模擬)
    A.CoarseDropout(max_holes=3, max_height=6, max_width=6, p=0.3),
    
    A.Resize(32, 128),
    A.Normalize(mean=[0.5], std=[0.5]),
])
```

質感劣化系を**厚めに**して、デプロイ環境の画質バリエーションを再現する。

### データフォーマット (2段階)

**Stage 1: 元画像 + アノテーション** (1画像 = 1JSON、Git管理):

```json
// annotations/img_001.json
{
  "image": "samples/img_001.jpg",
  "image_size": [4000, 3000],
  "source_video": "videos/v1.mp4",   // 動画由来の場合
  "labels": [
    {
      "id": 0,
      "bbox": [120, 340, 480, 420],
      "text_bbox": [180, 360, 460, 410],  // シリアル文字領域 (書き換え用)
      "text": "E300MM000032",
      "confidence": 0.99,
      "is_clear": true
    }
  ]
}
```

**Stage 2: 認識訓練用クロップ** (Stage 1 から自動生成、.gitignore):

```
data/recognition/
├── train/
│   ├── real/         # 実画像クロップ
│   ├── replaced/     # テキスト書き換え版
│   └── synthetic/    # TRDG 合成 (任意)
├── val/
├── test/             # 実画像のみ、書き換え版禁止
└── labels.tsv
```

### テストセット隔離 (重要)

```
test set: 動画とは別セッションで撮影した実写真 50枚
  - 訓練に使った動画と日時・場所を分ける
  - テキスト書き換え版は一切混入させない
  - Claude Code にも見せない (公平な評価のため)
  - 最終評価でのみ使用
```

これがないとモデルが「自分の合成画像で良いスコアを出す」だけになり、実環境性能を測れない。

### 必要なデータ量の目安

| 撮影努力 | データ生成 | 期待 CER | 段階 |
|---|---|---|---|
| 動画 3本 + 写真 30枚 | 200フレーム + 書き換え 2000枚 | 1〜2% | PoC |
| 動画 5本 + 写真 50枚 | 400フレーム + 書き換え 5000枚 | 0.5〜1% | MVP |
| 動画 10本 + 写真 100枚 | 800フレーム + 書き換え 10000枚 | < 0.5% | プロダクション |

**動画 5〜10本撮影 (撮影時間 30〜60分) でプロダクション品質に到達可能**。
撮影端末は 2〜3 種類を混ぜる。製造バッチも 5種類以上が望ましい。

### 失敗ケース追加収集ループ

```
1. 訓練 → 評価
2. test で失敗したケースを分析 (どの文字パターン、どの条件)
3. その条件を狙った動画を1本追加撮影 (5分)
4. 抽出 → ラベル → 書き換え → 再訓練
```

無計画に撮影量を増やすより、**失敗パターンを狙い撃ち**が圧倒的に効率的。

---

## 5. 技術スタック・リポジトリ構成

### 訓練側 (Python)

- Python 3.10+, PyTorch 2.x
- Albumentations (データ拡張)
- TRDG (補助的な合成データ生成、任意)
- onnx, onnx-simplifier, onnxruntime (エクスポート/量子化)
- OpenCV (bbox refinement)
- pytest

ラベリングは Claude Code セッションで実行するため、anthropic SDK は不要。

### 推論側 (TypeScript)

- TypeScript 5.x
- `onnxruntime-web` (WebGPU + WASM)
- `opencv-js` (古典CV検出用、既存環境のものを流用可能)
- Vite (ビルド) / Vitest (テスト)
- 出力: ESM + CJS

### リポジトリ構成

pnpm workspace モノレポ:

```
meiban-ocr/
├── HANDOFF.md
├── CLAUDE.md
├── LABELING.md
├── README.md
├── LICENSE
├── pnpm-workspace.yaml
├── videos/                    # 生動画 (.gitignore推奨)
├── samples/                   # 抽出フレーム + 直接撮影 (.gitignore)
├── samples_test/              # テスト用実画像 (隔離、Git管理 or 別管理)
├── annotations/               # Stage 1 アノテーション (Git管理)
│   ├── img_001.json
│   ├── _report.md
│   └── ...
├── data/                      # Stage 2 訓練データ (.gitignore)
│   └── recognition/
│       ├── train/{real,replaced,synthetic}/
│       ├── val/
│       ├── test/
│       └── labels.tsv
├── fonts/                     # OCR-A, OCR-B 等 (テキスト書き換え用)
├── packages/
│   ├── trainer/               # Python 訓練
│   │   ├── pyproject.toml
│   │   ├── configs/
│   │   ├── src/meiban_ocr_trainer/
│   │   │   ├── data/
│   │   │   │   ├── extract_frames.py   # 動画 → フレーム
│   │   │   │   ├── extract_crops.py    # Stage 1 → Stage 2
│   │   │   │   ├── text_replace.py     # テキスト書き換え
│   │   │   │   ├── refine_bbox.py      # bbox 精密化
│   │   │   │   ├── dataset.py
│   │   │   │   ├── augment.py
│   │   │   │   └── synth.py            # 補助合成データ
│   │   │   ├── models/tiny_ocr.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── export.py
│   │   └── tests/
│   └── runtime/               # TypeScript 推論 (npm公開)
│       ├── package.json
│       ├── src/
│       │   ├── index.ts
│       │   ├── MeibanOCR.ts
│       │   ├── detector.ts
│       │   ├── recognizer.ts
│       │   ├── preprocess.ts
│       │   ├── decoder.ts
│       │   ├── vendors.ts
│       │   └── constants.ts
│       ├── tests/
│       └── examples/react-camera/
├── models/                    # ONNX (Git LFS or HuggingFace)
└── benchmark/
```

---

## 6. 実装フェーズ

### Phase 1: ベースライン (Week 1〜2)

**ゴール**: 動画ベースのデータパイプラインが回り、CRNN 訓練が完了、評価セットで CER < 2%。

タスク:
- [ ] リポジトリ初期化 (pnpm workspace + Python project)
- [ ] ユーザーから動画 5本 + 直接撮影写真 30〜50枚 + test用画像 50枚を受領
- [ ] `extract_frames.py` 実装、動画から 1fps でフレーム抽出
- [ ] `samples/` に抽出フレームと直接撮影を集約
- [ ] Claude Code で `LABELING.md` を実行、`annotations/*.json` を生成
- [ ] `extract_crops.py` 実装、Stage 1 → Stage 2 変換
- [ ] `text_replace.py` 実装、各クロップから10〜30バリエーション生成
- [ ] 補助合成データ生成 (`synth.py`、任意)
- [ ] PyTorch Dataset/DataLoader (実/書換/合成のミックス対応)
- [ ] `TinyOCRModel` 実装
- [ ] 訓練・評価スクリプト (`train.py`, `evaluate.py`)
- [ ] **test セット (`samples_test/`) で最終評価**

**DoD**:
- 評価セットで CER < 2.0%
- CER / WER / 完全一致率がレポートされる
- データパイプライン全体が `make data` 等で再現可能
- ユニットテスト pass

### Phase 1.5: 失敗ケース追加収集 (任意)

Phase 1 の結果次第で実施。失敗パターンを分析し、その条件を狙った動画を1〜2本追加撮影。
再パイプライン実行で CER < 1% を目指す。

### Phase 2: ONNX化 + 軽量化 (Week 3)

**ゴール**: 2MB以下、推論20ms以下のONNXモデル。

タスク:
- [ ] PyTorch → ONNX エクスポート (`export.py`)
- [ ] onnx-simplifier で簡略化
- [ ] 動的量子化 (INT8)
- [ ] Python onnxruntime での精度確認 (量子化前後の比較)
- [ ] opset_version=17 (onnxruntime-web互換)

**DoD**:
- モデルサイズ < 2MB
- INT8 量子化後の CER 劣化 < 0.2%
- ONNX が onnxruntime-web で読み込める

### Phase 3: ブラウザランタイム (Week 4〜5)

**ゴール**: npm パッケージとして使えるライブラリ。

タスク:
- [ ] `MeibanOCR` クラス実装 (init / recognize / destroy)
- [ ] 古典CV検出 (`detector.ts`) — OpenCV.js でラベル矩形抽出
- [ ] 前処理関数 (Canvas → Float32Array)
- [ ] CTC Greedyデコーダ
- [ ] **6段階補正パイプライン** (backend `PlateSerialNumber.php` と互換)
- [ ] ベンダーパターン定義の外部化 (Ericsson + 将来追加可能な構造)
- [ ] WebGPU / WASM フォールバック
- [ ] パターンマッチ機能 (`recognize(img, { vendor: 'ericsson' })`)
- [ ] サンプルReactアプリ (`examples/react-camera/`)
- [ ] Vitest ユニットテスト + Playwright E2E

**API設計**:

```typescript
const ocr = new MeibanOCR({
  modelUrl: 'https://cdn.example.com/tiny-ocr-v1.onnx',
  executionProviders: ['webgpu', 'wasm'],
});

await ocr.init();

// 単一画像から複数コード抽出 (vendor指定でパターン制約 + 6段階補正適用)
const results = await ocr.recognize(canvas, {
  vendor: 'ericsson',          // 内部で strictRegex / partialRegex を適用
  minConfidence: 0.7,
});

// results: [{ text, confidence, bbox, matchStage }, ...]
// matchStage: どの段階でマッチしたか (1〜6, または null = 未マッチ)
```

**DoD**:
- npm install で使える
- WebGPU で全体 < 100ms (検出+認識合計、ラベル20個)
- iOS Safari + Android Chrome 動作確認
- サンプルアプリが動く

### Phase 4: 実データFine-tune + 公開 (Week 6〜7)

**ゴール**: 実環境でコード完全一致率 > 98%、公開。

タスク:
- [ ] 実画像データの追加収集 (難しいケース中心、500枚以上)
- [ ] Claude APIで自動ラベル → 信頼度低い分を目視確認
- [ ] Fine-tuning 実行
- [ ] Tesseract.js との A/B 比較レポート
- [ ] HuggingFace Hub にモデル公開
- [ ] npm v1.0.0 公開 (Apache 2.0)
- [ ] README + サンプル整備

**DoD**:
- 実画像評価セットで完全一致率 > 98%
- HuggingFace Hub + npm 公開完了

---

## 7. ルール

### やってほしいこと

- 各 Phase の DoD を満たすまで次に進まない
- コミットは小さく頻繁に (1機能1コミット)
- 重要な決定はコメントに `# Why: ...` で理由を残す
- 既存の参考実装 (セクション9) を活用
- テストは複雑なロジックに対しては書く

### やらないでほしいこと

- **未使用の汎用化・過剰な抽象化** (YAGNI)
- **多言語対応の先取り** (Phase 4 までは英数字のみ)
- **GPU前提の実装** (CPUでも訓練可能に)
- **巨大なコミット/PR**
- **GPL系コードのコピー** (Apache 2.0 / MIT / BSD のみ参考可)

### 要相談 (ユーザー確認必須)

- アーキテクチャの根本変更 (CRNN → 別モデル等)
- 公開APIの破壊的変更
- 依存パッケージの追加 (`onnxruntime-web`, PyTorch標準, anthropic, Albumentations 以外)
- ライセンス選択の変更

### 報告タイミング

- 各 Phase の DoD 達成時
- 設計判断が必要な時
- 2時間以上詰まった時 (試したことと選択肢を整理して相談)
- 想定外の発見があった時

---

## 8. 評価指標

```python
from torchmetrics.text import CharErrorRate

cer = CharErrorRate()           # 文字単位の編集距離 (主要指標)
```

評価項目:
- **CER**: 文字単位の編集距離
- **WER**: 単語単位の編集距離
- **EM (Exact Match)**: コード全体が完全一致した割合
- **推論時間**: 検出 + 認識の合計 (Webブラウザ実測)
- **モデルサイズ**: ONNXファイルサイズ

ベンチマークスクリプトを `benchmark/` に置き、CI で実行可能にする。

---

## 9. 参考実装

ゼロから書かず、以下を参考にする (ライセンス確認必須):

| プロジェクト | 用途 | ライセンス |
|---|---|---|
| `clovaai/deep-text-recognition-benchmark` | CRNN実装の基準 | Apache 2.0 |
| `baudm/parseq` | モダンな認識モデル | Apache 2.0 |
| `ndl-lab/ndlocr-lite` | ブラウザ実装の参考 | CC BY 4.0 |
| Anthropic Cookbook (vision) | Claude API使用例 | MIT |

---

## 10. 初期タスクリスト (Week 1)

### Day 1: セットアップ + 素材受領
1. リポジトリ作成、Apache 2.0 LICENSE、.gitignore
2. pnpm workspace 設定
3. `packages/trainer/` Python project (uv or poetry)
4. `packages/runtime/` TypeScript project (Vite library mode)
5. ユーザーから動画 5本 + 直接撮影 30〜50枚 + test用 50枚を受領
6. `fonts/` に OCR-A, OCR-B 等を配置
7. **動作確認**: `pnpm install` と Python の sync が通る

### Day 2: 動画フレーム抽出 + ラベリング
1. `extract_frames.py` 実装、5本の動画から抽出 (合計 200〜500フレーム)
2. 抽出結果を `samples/` に集約 (フレーム + 直接撮影)
3. Claude Code セッションで `LABELING.md` を実行
4. `annotations/_report.md` の確認
5. **動作確認**: 全画像の JSON が生成され、警告件数を把握

### Day 3: データパイプライン構築
1. `extract_crops.py` 実装、Stage 1 → Stage 2 変換
2. クロップ画像の目視確認 (10枚程度サンプル)
3. `text_replace.py` 実装、各クロップから10〜30バリエーション生成
4. (任意) `synth.py` で補助合成データ
5. **動作確認**: `data/recognition/` に train/val/test が揃う

### Day 4〜5: モデル実装
1. `TinyOCRModel` 実装
2. CTC tokenizer (文字列⇔ID)
3. PyTorch Dataset / DataLoader (augmentation 込み)
4. ユニットテスト
5. **動作確認**: ダミーデータで訓練ループが回る

### Day 6〜7: 訓練・評価
1. `train.py` (config駆動)
2. `evaluate.py` (CER / WER / EM)
3. 訓練実行 → val 監視
4. **test セット (samples_test/)** で最終評価
5. **Phase 1 DoD 確認**: CER < 2%

Week 1 終了時にユーザーへ進捗報告 + Phase 1.5 か Phase 2 開始判断。

---

## 11. 補足

### Phase 1 着手前に必要なもの (ユーザー提供)

- **動画 5〜10本** (各1分程度、計 5〜10分)
  - シーン1〜3: ラベルシートを接写でスイープ
  - シーン4〜5: 引きで全体撮影
  - 別バッチのラベルを混ぜる
  - 撮影端末を 2〜3 種類混ぜる
- **直接撮影写真 30〜50枚** (動画では取りにくいシーン用)
- **test 用画像 50枚** (`samples_test/`、訓練に絶対使わない隔離セット)
  - 動画とは別セッション・別日に撮影が望ましい
  - 製品出荷判定のゴールドスタンダード
- **OCR-A / OCR-B フォント** (テキスト書き換え用、Open Font License)
- Claude Code が動作する環境 (Claude サブスク)

### ライセンス・コードコメント

- 公開API・README・コミットメッセージ: 英語
- 内部のコメント・ドキュメント: 日本語OK (技術用語は英語のまま)

### Git運用

- メインブランチ: `main`
- ブランチ運用: 各 Phase ごとに `phase-1`, `phase-2` のブランチを切って PR ベース
- コミットメッセージ: Conventional Commits 推奨 (`feat:`, `fix:`, `refactor:` 等)

---

質問・不明点があれば作業開始前に必ず確認すること。
