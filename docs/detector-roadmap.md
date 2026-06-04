# 検出器ロードマップ — full-frame (reticle) と 古典CV (数枚)

> 2026-06-04 作成。 KGI v2 (reticle 1枚=必達 / 数枚=ストレッチ) に対応する検出器方針。
> 試作・実測 (`scripts/diag_cv_detector.py`, `diag_cv_lines.py`, `diag_model_compare.py`) に基づく。

## 1. 結論サマリ

| 用途 | 検出器 | 状態 | 実測 |
|---|---|---|---|
| **reticle 1枚 (必達)** | full-frame `[[0,0,w,h]]` | ✅ 即採用可 | v7-crnn full-frame EM **89%** / 15ms |
| **数枚同時 (ストレッチ)** | 古典CV text-line 抽出 → 各行を認識 | 🔄 原理実証済・要チューニング | 検出行を捕まえれば conf 0.94+ で正解 |

配信モデルは **v7-crnn (ctc) 一択** (fixed-head は full-frame EM 10.7% で不可)。

## 2. 銘板の実態 (試作で判明)

エリクソン銘板は **白ラベル上の4行構成**:

```
Radio 4471HP 81
製造番号  E325MM500327     ← 欲しいのはこの行のシリアル部
2024年元月
エリクソン・ジャパン株式会社
```

- シリアル行は **「製造番号」+ シリアル** が同一行。
- ラベルは **トレイ上で傾き・透視ゆがみ** がある。
- 動画 (10_44 等) は **38枚を一度に映すのではなく、 カメラがパンして舐める** 撮影 → 1 frame の可読枚数は数枚 (best frame で 6/38)。

## 3. 試作で確かめたこと

1. **複数行ブロックを掴むと読めない** (0回収)。 32×128 に潰すと潰れる。
2. **1 text-line を捕まえれば読める**: 行全体 crop (「製造番号 E325MM…」込み) を v7-crnn に渡すと
   **conf 0.94〜0.95 で正解** (3/38、 best frame)。 → recognizer はプレフィックスを許容。
3. **行を下手に右だけ切ると 0**: シリアル部だけ切り出そうとせず **行を丸ごと** 渡すのが正解。
4. 現状 recall が sliding-window (6) より低い (3) のは **ラベルの傾き** が主因
   (`boundingRect` が斜め文字に弱く crop がゆるむ)。 ただし候補 26 個 vs 3430 窓で **~130x 速い**。

## 4. 古典CV 検出アルゴリズム (試作版)

```
gray → blackhat (暗刻印/印字を強調) → Otsu 2値化
     → 横長 close (文字を1行に連結)
     → findContours → aspect 3〜14:1 / サイズで text-line 候補に絞る
     → 各候補を crop → v7-crnn 認識 → regex フィルタ (= 強力な最終フィルタ)
```

recognizer + Ericsson regex が「シリアル行か否か」の最終判定を兼ねるので、
検出は **over-generate (行を多めに出す) で良い**。 誤検出行は regex で落ちる。

## 5. ストレッチ達成に向けた改善 (優先順)

| # | 改善 | 期待効果 |
|---|---|---|
| 1 | `minAreaRect` + 透視補正で **傾いたラベル行を deskew** してから crop | 傾き起因の取りこぼし解消 (recall 主因) |
| 2 | 白ラベル領域を先に検出 → ラベル内に限定して行抽出 | 背景ノイズ除去・候補数削減 |
| 3 | frame 間 consensus (同一 plate を複数 frame で多数決) | 5/6 混同の transient 誤読除去 = **FP=0 の本体** |
| 4 | (任意) recognizer を「行レベル crop」で再訓練 | 製造番号プレフィックス込みの頑健化 |

## 6. OpenCV.js への移植方針

- 検出器は `DetectorFn: (ImageData) => BBox[]` に適合 (`packages/runtime/src/detectors/types.ts`)。
  Python 試作の cv2 呼び出しは OpenCV.js にほぼ1:1対応 (`cvtColor` / `morphologyEx` /
  `threshold` / `findContours` / `boundingRect` / `minAreaRect`)。
- 既存 `sliding-window.ts` と同じ場所に `classic-cv.ts` を追加し、 `createClassicCvDetector()` を export。
- runtime flags: full-frame・古典CV いずれも `prefilter: false`, `recenter: false` 推奨
  (sliding-window 前提の処理を切る)。
- reticle (full-frame) は OpenCV.js すら不要 = `detector: (img) => [[0,0,img.width,img.height]]`。

## 7. 関連

- `scripts/diag_model_compare.py` — モデル選定 (v6 vs v7-crnn, full-frame EM)
- `scripts/diag_cv_detector.py` / `diag_cv_lines.py` — 古典CV 検出試作・行レベル検証
- `scripts/diag_firing.py` — sliding-window 発火ゼロの切り分け
- `docs/KGI.md` — KGI v2
- memory: `model-selection-crnn`, `kgi-scope-relaxed`, `session-zero-firing-cause`
