"""文字セット定義。TypeScript側 (`packages/runtime/src/constants.ts`) と同期させる。"""

CHARSET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 chars, index 0..35
BLANK_IDX: int = 36  # CTC blank
NUM_CLASSES: int = 37

# ===== 12-head 固定長アーキテクチャ (Phase 2b) =====
# Ericsson serial は厳密に 12 文字、使う文字種は 0-9 + E + M の **12 種のみ**。
# CRNN+CTC は汎用 OCR の構造で「文字何でも + blank」だが、固定長 13 クラスに絞ることで:
#   1. パラメータ削減 (Linear 出力 37 → 13)
#   2. ∅ クラスを位置毎に持たせて構造的に reject 表現可能 (hallucination 防止)
#   3. 出力解釈が単純 (CTC decode 不要、argmax だけ)
CHARSET_12H: str = "0123456789EM"  # 12 chars, indices 0..11
EMPTY_IDX: int = 12  # ∅ class (no character at this position)
NUM_CLASSES_12H: int = 13
FIXED_LENGTH: int = 12  # Ericsson serial length

# 入力サイズ (CRNN)
INPUT_HEIGHT: int = 32
INPUT_WIDTH: int = 128

# 正規化 (グレースケール、[0,1] → mean/std で標準化)
NORM_MEAN: tuple[float] = (0.5,)
NORM_STD: tuple[float] = (0.5,)
