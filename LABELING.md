# LABELING.md — Claude Code 用ラベリング指示書

このドキュメントは、`samples/` 内の画像から訓練データを生成するための
Claude Code 用の作業指示書です。

## このファイルの使い方

Claude Code セッションで以下のようにユーザーが指示します:

> `LABELING.md` を読んで、その手順に従って `samples/` 内の全画像をラベリングしてください。

Claude Code はこの指示書を読み、以下の手順を実行します。

---

## 作業の目的

`samples/` 内の銘板画像から、**2種類の領域**を抽出して `annotations/` に保存します。

1. **positive region**: Ericsson シリアルコードとその位置
2. **negative region**: 「コードに見えるがコードではない」領域 (= reject 訓練データ)

negative は CRNN モデルに「何も読み取らない」判断を学習させるための重要なサンプル
です。これが無いと、モデルは常に何かしらのテキストを出力してしまい、検出器の
誤検出をそのまま受け入れる事故が起きます。

---

## 対象コード仕様

### Ericsson 銘板のシリアル (positive)

- **厳格パターン**: `/^E[39]\d{2}MM\d{6}$/`
- **部分一致パターン**: `/E[39]\d{2}MM\d{6}/`
- フォーマット: `E` + (`3` or `9`) + 数字2桁 + `MM` + 数字6桁
- 例: `E300MM000032`, `E300MM999022`

### 重要な観察ポイント

- 銘板は金属光沢の矩形ラベル
- 1画像に複数のラベルが写っていることが多い (グリッド配置)
- 「製造番号」のラベルが直前にあり、その右側にシリアルが印字されている

### negative の候補 (= コードではないがコードと混同しうる領域)

| subkind | 例 |
|---|---|
| `other_text` | 警告ラベル、型番 (例: `RRUS 11 B1`), 日付, 会社名, バーコード |
| `background` | 金属面・配線・空き領域 (テキスト無し) |
| `partial` | シリアルの一部が遮蔽・反射で欠けた領域 (「読めるが完全でない」) |
| `other_vendor` | 別ベンダーのコード (例: Huawei `WS6800-...`) |
| `mined` | 訓練済みモデルが誤って pattern 合致を出した領域 (hard negative mining 由来) |

---

## 作業手順

### Step 1: 画像リストの取得

```bash
ls samples/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null
```

すべての画像ファイルをリストアップしてください。

### Step 2: 各画像のラベリング

各画像について、以下を実行:

1. `view` ツールで画像を表示
2. 画像内のすべての Ericsson 銘板ラベル (positive) を特定
3. 画像内の **negative 候補領域**を特定 (上記表の subkind を参考に、1画像あたり 3〜10件目安)
4. 各 region について以下を判定:

**positive**:
- **text**: 読み取ったシリアル文字列 (12文字)
- **bbox**: ラベル全体 (金属プレート全体) の境界ボックス `[x1, y1, x2, y2]`
- **text_bbox**: シリアル文字列**だけ**を囲む境界ボックス (テキスト書き換え用、重要)
- **quality**: `"clear"` (明瞭) / `"blur"` (ブレ・反射で不鮮明) / `"partial"` / `"occluded"`
- **confidence**: 読み取りの自己評価 (0.0 〜 1.0)

**negative**:
- **bbox**: 切り出す領域 (高さは 25〜80px、幅 80〜400px 程度を目安)
- **subkind**: `"other_text"` / `"background"` / `"partial"` / `"other_vendor"`
- **text_visible** (optional): その領域に見えるテキスト (参考。学習には使わない)

5. 結果を `annotations/{basename}.json` に保存 (v2 schema、下記)

### Step 3: スキーマ (v2: `regions[]`)

```json
{
  "image": "img_001.jpg",
  "image_size": [4000, 3000],
  "source_video": "videos/v1.mp4",
  "vendor": "ericsson",
  "schema_version": 2,
  "regions": [
    {
      "id": 0,
      "category": "positive",
      "bbox": [120, 340, 480, 420],
      "text_bbox": [180, 360, 460, 410],
      "text": "E300MM000032",
      "vendor": "ericsson",
      "quality": "clear",
      "confidence": 0.99,
      "claude_verified": true
    },
    {
      "id": 1,
      "category": "positive",
      "bbox": [520, 340, 880, 420],
      "text_bbox": [580, 360, 860, 410],
      "text": "E300MM000033",
      "vendor": "ericsson",
      "quality": "clear",
      "confidence": 0.98,
      "claude_verified": true
    },
    {
      "id": 2,
      "category": "negative",
      "bbox": [100, 80, 400, 130],
      "subkind": "other_text",
      "text": "",
      "text_visible": "RRUS 11 B1",
      "claude_verified": true
    },
    {
      "id": 3,
      "category": "negative",
      "bbox": [1200, 900, 1500, 950],
      "subkind": "background",
      "text": "",
      "claude_verified": true
    }
  ]
}
```

**フィールド説明**:

| field | positive | negative | 説明 |
|---|:-:|:-:|---|
| `category` | ✓ | ✓ | `"positive"` または `"negative"` |
| `bbox` | ✓ | ✓ | 切り出す領域。positive はパディング込みのラベル全体、negative は切り出したい領域そのまま |
| `text_bbox` | ✓ | — | シリアル文字列だけの bbox (text_replace 用) |
| `text` | ✓ | `""` 固定 | positive はシリアル文字列、negative は必ず空文字 |
| `vendor` | optional | — | 画像 vendor の override (通常は省略) |
| `quality` | optional | — | `"clear"` / `"blur"` / `"partial"` / `"occluded"` |
| `confidence` | ✓ | — | ラベリング時の自己評価 |
| `subkind` | — | ✓ | `"background"` / `"other_text"` / `"partial"` / `"other_vendor"` / `"mined"` |
| `text_visible` | — | optional | negative 領域に見えるテキスト (参考メモ) |
| `claude_verified` | ✓ | ✓ | VLM による目視確認済みなら `true` |

> **互換性**: 旧 v1 schema (`labels[]` のみ、positive のみ) で書かれた annotation は
> Python 側 loader (`meiban_ocr_trainer.data.annotation.load_annotation`) が自動的に
> `regions[]` (全部 `category: "positive"`) に変換して読み込みます。新規ファイルは
> 必ず v2 で書いてください。

### Step 4: サマリレポート生成

全画像の処理が完了したら、`annotations/_report.md` に統計サマリを保存:

```markdown
# Labeling Report

- 処理画像数: N
- positive 総数: P
- negative 総数: Q  (内訳: background X / other_text Y / partial Z / other_vendor W)
- 平均 positive 数/画像: P/N
- 平均 negative 数/画像: Q/N

## positive 信頼度分布
- confidence >= 0.95: X件 (XX%)
- 0.80 <= confidence < 0.95: Y件
- confidence < 0.80: Z件

## positive quality 分布
- clear: A件
- blur: B件 (要確認)

## 警告
- パターンに一致しないシリアル: (リスト)
- bbox が画像外に出ている: (リスト)
- 重複しているシリアル: (リスト)

## サンプル別 region 数 (上位/下位)
- img_005.jpg: 24 pos / 8 neg
- img_001.jpg: 20 pos / 5 neg
- ...
```

---

## 重要な指針

### bbox の決め方

**positive**:
- ラベルの**金属部分全体**を含む矩形
- ラベル端から 2〜5px の余白
- 完全なピクセル精度は不要 (後で OpenCV で refine する)

**negative**:
- positive と**重ならない**こと (同じ領域を pos/neg 両方にすると CTC が壊れる)
- 1辺 50px 以下の極端に小さい領域は避ける (crop しても情報が無い)
- アスペクト比は positive と似た縦長 (高さ:幅 = 1:2〜1:8) が望ましい
  - 認識器の入力 32×128 に近いほど学習効率が高い

### negative の選び方 (重要)

**良い negative**:
- 検出器 (OpenCV.js) が誤って「ラベル候補」として拾いそうな矩形領域
- 文字や境界線がある (「コードっぽいが違う」)
- 同じ画像の別の部位 (背景・配線・別のラベル)

**避けるべき negative**:
- positive の隣接領域で部分的にシリアル文字を含む (CTC 訓練が混乱)
- 画像の隅の極端な低テクスチャ (学習に寄与しない)
- positive と全く同じテクスチャの繰り返し (情報量が低い)

### confidence の付け方 (positive のみ)

- **0.95 以上**: 全文字が明瞭に読める、誤読の可能性なし
- **0.80〜0.95**: ほぼ確実だが一部の文字に微妙な不確かさ
- **0.50〜0.80**: 推測を含む読み取り
- **0.50 未満**: 読めない or 大きな推測 → `quality: "blur"` も併用

### quality の判定 (positive のみ)

- `"clear"`: 全12文字明瞭、紛らわしい文字 (O/0, I/1 等) の判別も明確
- `"blur"`: ブレ・反射・部分欠けで一部不明瞭
- `"partial"`: 一部の文字が画像外に切れている
- `"occluded"`: 何かに隠れている (ジップ袋等)
- `blur` / `partial` / `occluded` は extract_crops で skip される (claude_verified=false 扱い)

### パターン違反への対応 (positive)

検出したテキストが Ericsson の厳格パターン `/^E[39]\d{2}MM\d{6}$/` に一致しない場合:

1. **読み取り誤りの可能性が高い** → confidence を下げる、quality を `"blur"` に
2. ただしテキスト自体は推測したものを記録する (補正なし、生の読み取りを残す)
3. `_report.md` の警告セクションに記録

### スキップの判断

以下の画像は処理対象外として `annotations/_skipped.json` に理由付きで記録:

- 画像が壊れている / 読み込めない
- 銘板が1つも写っていない (negative も収集できない場合)
- 完全にブレていて読み取り不可

---

## negative の最小目安

| シーン | positive 数 | negative 推奨数 |
|---|:-:|:-:|
| 銘板グリッド (10件以上) | 10〜20 | 5〜10 |
| 銘板単体 + 周辺機器 | 1〜3 | 5〜15 (周辺の文字や背景を多めに) |
| 銘板無し (背景のみ画像) | 0 | 5〜20 |

**val/test 用画像は最優先で negative を多めに** (reject 性能を測る base になる)。

---

## 進捗報告

50枚処理ごとに簡潔な進捗報告をしてください:

> 50/100 完了。positive 950件、negative 320件、平均 confidence: 0.96。
> パターン違反: 2件。続行します。

全件完了後、`annotations/_report.md` を生成し、ユーザーに完了報告。

---

## トラブルシューティング

### bbox がずれる気がする

VLM の bbox 精度は ±10〜30px 程度なので、ピクセル精度は不要。
後段の `refine_bbox.py` (OpenCV) で positive のみ精密化される。
negative は refine されないので、最初から余裕を持って bbox を取る。

### 画像が大きすぎる

そのままの座標で記録してください。リサイズ等は後段で行います。

### 画像内のラベルが 20 件超ある

すべて検出してください。1JSON あたりの region 数に上限はありません。

### `MM` 以外のフォーマットが見える (`MR`, `MN` 等)

Ericsson の正規パターンに違反するため、誤読の可能性が高い。
**判断:**
- 「シリアルだが読み間違えた」と判断できる → positive、quality `"blur"`
- 「別のコード/型番だ」と判断できる → **negative の `other_text` として記録**

### 別ベンダーのコード (Huawei 等) が見える

`category: "negative"`, `subkind: "other_vendor"` で記録。
`text_visible` に実際のコードを書いておくと debug 時に便利。

---

## 注意事項

- このタスクは**1〜2時間程度の長時間セッション**になります。途中で進捗を区切ってください
- 同じ画像を2回処理しないよう、既に `annotations/{basename}.json` が存在する画像はスキップ
- `samples/` を変更しないこと (読み取り専用)
- `annotations/` 以下のみ書き込み可
- **`samples_test/` には絶対手を入れない** (テスト隔離。負例追加もここではしない)

---

## RapidOCR 自動ラベリングとの併用

`packages/trainer/src/meiban_ocr_trainer/data/auto_label.py` は positive のみを
RapidOCR で自動抽出します (v2 schema で出力)。Claude Code の役割は:

1. auto_label が生成した `annotations/img_*.json` の positive を VLM で**目視検証**
   (合致したら `claude_verified: true` を立てる)
2. **negative を追加**して `regions[]` に append
3. `_report.md` を更新

新規撮影画像の場合は Claude Code が positive も含めて全て作成。
