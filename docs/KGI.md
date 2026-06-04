# KGI / Hard Targets — 倉庫リアルタイム検出の成功条件

> 2026-06-02 制定。 2026-06-04 **大改訂 (v2)**: 用途を「40枚一括バッチ」から
> 「reticle で 1 枚ずつ (数枚同時はストレッチ)」へ転換。 旧 §3 の per-frame q 逆算は退役。
>
> Hard Target が変わったら同ドキュメントを更新、 git で変更履歴を残す。

---

## 0. 改訂履歴

| 日付 | 版 | 変更点 |
|---|---|---|
| 2026-06-02 | v1 | 初版。 40枚一括バッチ前提で per-frame q ≥56% 等を逆算 |
| 2026-06-04 | v2 | **40枚一括を退役**。 reticle 1枚=必達 / 数枚=ストレッチへ。 検出器を sliding-window から古典CV回帰へ。 配信モデルを v7-crnn に確定 |

## 1. 用途の前提 (v2 改訂)

- 倉庫オペレータがスマホの **reticle (照準枠) に銘板を 1 枚収めて** スキャン
- ブラウザ edge OCR、 URANUS2 統合、 連続 fps で枠内を認識
- ground truth は「Claude Code / 高精度 OCR が読める値」、 読めなければ「答え無し」 (= 検出無しでよい)
- **40 枚一括は狙わない** (2026-06-04 ユーザー判断)。 一括で数枚〜10枚読めれば上出来 (ストレッチ)

### なぜ転換したか (実測根拠)

- reticle はユーザーが 1 枚を枠に収める UX。 **検出器不要 = reticle crop 全体を 1 窓として認識器へ直接渡す (full-frame detector)** のが正解。
- 旧 sliding-window は「検出器の代用品」だったが、 タイト crop では窓を surface できず発火せず、 フルシートでは 1363 窓 × 14秒/frame で破綻していた (URANUS2 実測)。
- full-frame detector は **発火 + 15ms/frame** を同時達成 (URANUS2 wasm 実測)。

## 2. Primary KGI (Hard Targets — 絶対線)

| 指標 | 目標 | 状態 (2026-06-04) |
|---|---|---|
| **誤読 accept 件数** | **= 0** (誤読の確定は絶対禁止) | ⚠️ 5/6混同が脅威 (§5) |
| **reticle 1 枚 read-rate** | **≥ 99%** (枠に正しく収めた銘板を確定できる) | 🔄 測定要 (full-frame EM 89% が下限) |
| **frame 全体時間** | **≤ 300ms** (OCR_INTERVAL 内) | ✅ 見込み (full-frame 15ms/wasm) |

### Why これらが絶対線か

- **0 誤読 accept**: 誤読が下流の倉庫システムに混入 → 検品ミス・誤出荷・在庫不整合。 1 件でも信用失墜。 `reject 優先方針` の絶対実装。
  - **注意 (2026-06-04)**: confidence では誤読を止められない (誤読も高 conf で出る、 §5)。 **frame 間 consensus が FP 防止の本体**。
- **reticle read-rate ≥ 99%**: ユーザーが枠に収めた 1 枚は、 数 frame 内にほぼ確実に確定する必要。 手入力に落ちる頻度を業務許容内に。
- **≤ 300ms**: ライブスキャンのリズムを保つ。

## 3. ストレッチ目標 (数枚同時)

必達ではないが、 達成できれば業務効率が大きく上がる:

| 指標 | 目標 |
|---|---|
| 1 frame で同時確定できる枚数 | 数枚 〜 **10 枚** |
| その時の frame 全体時間 | ≤ 300ms (検出 ~50ms + N×15ms) |

- 実現手段: **古典CV (OpenCV.js) で銘板矩形を抽出 → 各 crop を full-frame 認識** (§4)。
- 旧 v1 §3 の「per-frame q ≥56%」「40枚×10frameで100%読了」の逆算は **退役** (40枚を狙わないため)。

## 4. アーキテクチャ (v2 確定)

```
reticle / フレーム
   → 検出 (reticle: full-frame [[0,0,w,h]] / 数枚: 古典CV 矩形抽出)
   → 各ラベルを crop
   → CRNN (v7-crnn) で各 crop を認識
   → パターン補正 + regex フィルタ
   → frame 間 consensus (誤読除去・確定)
```

- **配信モデル = v7-crnn (ctc)**。 fixed-head 系は full-frame EM 10.7% で壊滅、 v7-crnn は 89.0% (2026-06-04 実測)。
- **sliding-window は廃止候補**。 検出器インターフェース (`DetectorFn`) は維持し、 full-frame / 古典CV を差し替える。
- runtime flags: full-frame 入力では `prefilter: false`, `recenter: false` 推奨 (sliding-window 前提の処理を切る)。

## 5. 既知の脅威・未解決 (P2)

| # | 項目 | 内容 | 対策方針 |
|---|---|---|---|
| P2 | **5/6 混同** (`E325→E326`) | v7-crnn でも残る。 誤読が **高 conf** で出る (誤読中央値 0.92、 conf≥0.5 を 100% 通過) → confidence で止められず **誤読 accept = FP** に直結 | (1) frame 間 consensus で transient な誤読を除去、 (2) 5/6 を分離する訓練データ強化 |
| — | **モデルサイズ** | v7-crnn 16.5MB (fp16) で **≤5MB KGI を超過**。 INT8 でも ~8MB | size KGI を緩和 (精度優先) or 小型 CRNN の再訓練/蒸留。 別途判断 |
| — | **評価ハーネス** | session_diagnose は GT を全動画から、 評価は先頭 3 秒のみ → 構造的に不整合 (10_44 が見かけ 0%) | GT/評価窓を reticle 前提で組み直す or 「完了まで」回す |

## 6. リソース目標

| 指標 | 目標 | 状態 |
|---|---|---|
| モデル合計 (認識のみ。 reticle は検出器不要) | ≤ 5MB | ❌ v7-crnn 16.5MB (§5) |
| メモリ peak (edge スマホ想定) | ≤ 200MB | ⚠️ 未計測 |
| frame 全体時間 (WebGPU) | ≤ 300ms | ✅ 見込み |

## 7. session-level メカニズム (consensus は FP 防止の本体)

per-frame の誤読を構造的に防ぐため、 runtime 側で実装する:

- **k-of-n consensus**: 同一 plate (reticle なら時間方向、 数枚なら bbox tracking) で k frame 以上同じ text が出たら確定。 1〜k-1 回は reject。
  - 5/6 混同のような transient 誤読は k 回連続しにくい → consensus で落ちる = **FP=0 の主防衛線**。
- **confidence 集約**: 表示用。 conf 単体は品質指標として信用しない (§5)。

## 8. このドキュメントの位置付け

- HANDOFF.md §1 「数値目標」 (CER < 0.5%, EM > 98%) は **モデル単独指標**。 本ドキュメントの reticle/session 指標と相補。 EM > 98% は現状 (v7-crnn full-frame 89%) 未達で、 5/6 混同 (§5) がボトルネック。
- 改訂時は git commit に「KGI 変更」と明記、 差分を本文 §0 に残す。
- 関連:
  - memory: `kgi-scope-relaxed` / `model-selection-crnn` / `session-zero-firing-cause`
  - `HANDOFF.md` — プロジェクト全体仕様
  - `packages/runtime/src/detectors/` — 検出器インターフェース (full-frame / 古典CV 差し替え点)
  - `packages/trainer/src/meiban_ocr_trainer/tools/session_diagnose.py` — session 評価 (要 reticle 前提改修)
