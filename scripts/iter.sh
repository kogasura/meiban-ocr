#!/usr/bin/env bash
# iter.sh — 訓練 → ONNX export → dist-uranus2 → URANUS2 sync を 1 コマンドで。
#
# 想定フロー:
#   1. ユーザが訓練を回す: python -m meiban_ocr_trainer.train_fixed_head ...
#      → runs/<TS>_fh/best.pt が出る
#   2. 本スクリプトを実行:
#      $ scripts/iter.sh
#      → 最新 checkpoint を自動検出
#      → meiban-ocr-real-v<N+1>.onnx として export
#      → dist-uranus2/ を生成
#      → URANUS2 client の public/assets/meiban-ocr/ にコピー
#   3. URANUS2 側で commit して動作確認
#
# 設計:
# - 既存 real-v<N> の最大 N + 1 で自動採番
# - URANUS2 パスは環境変数 URANUS2_CLIENT で上書き可
# - --no-sync で URANUS2 同期だけスキップ可能 (build のみ確認)
# - --cleanup-old N で古い real-v<N> を直近 N 個残して削除 (models/ 肥大化対策)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
URANUS2_CLIENT="${URANUS2_CLIENT:-$HOME/jdf-dev/uranus2/client}"
RUNS_DIR="$REPO_ROOT/runs"
MODELS_DIR="$REPO_ROOT/models"

# ===== ヘルパー =====

log() { printf '\033[36m[iter]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[iter:ERROR]\033[0m %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
Usage: scripts/iter.sh [OPTIONS]

訓練後 1 コマンドで dist-uranus2 生成 + URANUS2 同期まで完了する。

Options:
  --checkpoint PATH    PyTorch checkpoint (default: 最新 runs/*/best.pt)
  --label LABEL        ONNX export label (default: real-v<N+1>、 既存 real-v<N> + 1)
  --no-sync            URANUS2 へのコピーをスキップ (build まで)
  --cleanup-old N      古い real-v* を最新 N 個残して削除 (models/ 容量対策)
  -h, --help           このヘルプ

Environment:
  URANUS2_CLIENT       URANUS2 client repo path (default: $HOME/jdf-dev/uranus2/client)

Examples:
  scripts/iter.sh                              # 全自動: 最新 ckpt + 次 version + URANUS2 同期
  scripts/iter.sh --label real-v10             # version を明示
  scripts/iter.sh --no-sync                    # build まで (URANUS2 同期しない、 dist 確認用)
  scripts/iter.sh --cleanup-old 3              # 直近 3 個 real-v 残してそれ以前削除
  URANUS2_CLIENT=/other/path scripts/iter.sh   # 配置先を上書き
EOF
}

# ===== オプション parse =====

CKPT=""
LABEL=""
DO_SYNC=1
CLEANUP_KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --no-sync) DO_SYNC=0; shift ;;
    --cleanup-old) CLEANUP_KEEP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# ===== 1. checkpoint 決定 =====

if [ -z "$CKPT" ]; then
  if [ ! -d "$RUNS_DIR" ]; then
    err "runs/ not found at $RUNS_DIR — train first or specify --checkpoint"
    exit 1
  fi
  CKPT=$(find "$RUNS_DIR" -maxdepth 2 -name 'best.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | awk '{print $2}' || true)
  if [ -z "$CKPT" ]; then
    err "no best.pt under runs/*/ — train first or specify --checkpoint"
    exit 1
  fi
  log "auto-detected latest checkpoint: $CKPT"
fi
if [ ! -f "$CKPT" ]; then
  err "checkpoint not found: $CKPT"
  exit 1
fi

# ===== 2. label 決定 =====

if [ -z "$LABEL" ]; then
  MAX_V=""
  if [ -d "$MODELS_DIR" ]; then
    MAX_V=$(ls "$MODELS_DIR"/meiban-ocr-real-v*.onnx 2>/dev/null \
      | grep -oE 'real-v[0-9]+\.onnx$' \
      | grep -oE '[0-9]+' \
      | sort -n | tail -1 || true)
  fi
  NEXT_V=$(( ${MAX_V:-0} + 1 ))
  LABEL="real-v$NEXT_V"
  log "auto-assigned label: $LABEL (max existing: real-v${MAX_V:-none})"
fi

# ===== 3. Export =====

log "exporting $CKPT → models/meiban-ocr-$LABEL.onnx"
cd "$REPO_ROOT"
PYTHONPATH=packages/trainer/src python3 -m meiban_ocr_trainer.export \
  --checkpoint "$CKPT" \
  --output-dir models/ \
  --name "meiban-ocr-$LABEL"

if [ ! -f "$MODELS_DIR/meiban-ocr-$LABEL.onnx" ]; then
  err "export failed: models/meiban-ocr-$LABEL.onnx not created"
  exit 1
fi

# ===== 4. Build dist-uranus2 =====

log "building dist-uranus2/"
cd "$REPO_ROOT/packages/runtime"
pnpm build:uranus2

# ===== 5. URANUS2 sync =====

if [ "$DO_SYNC" -eq 1 ]; then
  if [ ! -d "$URANUS2_CLIENT" ]; then
    err "URANUS2 client not found: $URANUS2_CLIENT"
    err "set URANUS2_CLIENT env var or use --no-sync"
    exit 1
  fi
  TARGET="$URANUS2_CLIENT/public/assets/meiban-ocr"
  log "syncing dist-uranus2/ → $TARGET/"
  rm -rf "$TARGET"
  mkdir -p "$TARGET"
  cp -r "$REPO_ROOT/dist-uranus2/"* "$TARGET/"
fi

# ===== 6. Cleanup =====

if [ "$CLEANUP_KEEP" -gt 0 ]; then
  log "cleanup: keeping latest $CLEANUP_KEEP real-v<N> version(s)"
  cd "$MODELS_DIR"
  # 各 real-v<N> の base + fp32/fp16/sim/report の関連ファイルを一括削除
  ls meiban-ocr-real-v*.onnx 2>/dev/null \
    | grep -oE 'real-v[0-9]+' \
    | sort -uV \
    | head -n -"$CLEANUP_KEEP" \
    | while read v; do
        log "  removing $v files"
        rm -f "meiban-ocr-$v.onnx" \
              "meiban-ocr-$v.fp32.onnx" \
              "meiban-ocr-$v.fp16.onnx" \
              "meiban-ocr-$v.fp32.sim.onnx" \
              "meiban-ocr-$v.report.json"
      done
fi

# ===== 7. Summary =====

log ""
log "========== done =========="
log "  label:        $LABEL"
log "  model:        $MODELS_DIR/meiban-ocr-$LABEL.onnx"
log "  bundled into: $REPO_ROOT/dist-uranus2/model/custom/meiban-ocr-fixed-head.onnx"
if [ "$DO_SYNC" -eq 1 ]; then
  log "  uranus2:      $URANUS2_CLIENT/public/assets/meiban-ocr/"
  log ""
  log "Next steps in URANUS2 client:"
  log "  cd $URANUS2_CLIENT"
  log "  git add public/assets/meiban-ocr/"
  log "  git commit -m 'chore(ocr): update meiban-ocr to $LABEL'"
  log "  git push"
else
  log "  uranus2:      (--no-sync skipped)"
fi
