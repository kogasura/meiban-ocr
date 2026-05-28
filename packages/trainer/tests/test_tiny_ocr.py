import torch

from meiban_ocr_trainer.constants import INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES
from meiban_ocr_trainer.models import TinyOCRModel


def test_forward_shape_and_dtype():
    model = TinyOCRModel(pretrained=False).eval()
    x = torch.randn(2, 1, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        y = model(x)
    # 期待: (B=2, T, NUM_CLASSES). stride 調整で T は INPUT_WIDTH/4=32 になるはず。
    assert y.shape[0] == 2
    assert y.shape[2] == NUM_CLASSES
    assert y.dtype == torch.float32


def test_t_is_long_enough_for_12_chars():
    """CTC は概ね T >= 2L-1 が必要。L=12 想定なので T>=23 をチェック。"""
    model = TinyOCRModel(pretrained=False).eval()
    x = torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
    with torch.no_grad():
        y = model(x)
    assert y.shape[1] >= 23, f"T={y.shape[1]} is too small for 12-char CTC target"


def test_ctc_loss_runs():
    """合成データで 1 step CTCLoss が走ることを確認 (NaN/Inf にならない)。"""
    import torch.nn.functional as F

    model = TinyOCRModel(pretrained=False)
    x = torch.randn(2, 1, INPUT_HEIGHT, INPUT_WIDTH)
    logits = model(x)                       # (B, T, C)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    input_lengths = torch.full((2,), logits.shape[1], dtype=torch.long)
    loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=36, zero_infinity=True)
    assert torch.isfinite(loss).item()
