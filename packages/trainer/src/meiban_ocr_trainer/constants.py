"""文字セット定義。TypeScript側 (`packages/runtime/src/constants.ts`) と同期させる。"""

CHARSET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 chars, index 0..35
BLANK_IDX: int = 36  # CTC blank
NUM_CLASSES: int = 37

# 入力サイズ (CRNN)
INPUT_HEIGHT: int = 32
INPUT_WIDTH: int = 128

# 正規化 (グレースケール、[0,1] → mean/std で標準化)
NORM_MEAN: tuple[float] = (0.5,)
NORM_STD: tuple[float] = (0.5,)
