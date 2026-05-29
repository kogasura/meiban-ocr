"""Memorization gate: 訓練 summary.json の予測値に dummy 範囲外の serial が無いか検証。

Iter5 v5 #9 fix。訓練後の `runs/<run>/summary.json` には `test_samples` / `val_samples`
の `pred` がそのまま記録される。これが `E[39]\\d{2}MM\\d{6}` のうち `E300MM` 以外
であれば、モデルが本番 Ericsson 形式の serial を memorize した痕跡となり、その
checkpoint を ONNX export → npm 配信に乗せると **モデル経由で実シリアルが漏洩**
する経路が成立する (= Iter4 v4 #1 が想定したリスク)。

本ツールは export.py の冒頭から呼ばれ、ゲートとして機能する。CI で
`runs/<run>/summary.json` を後追い検査する補助 workflow としても使える。

Exit code:
  0: clean (no memorized prefix)
  1: memorization detected (sample preds 含む)
  2: usage / IO error

Usage:
    python -m meiban_ocr_trainer.tools.check_no_memorized_prefix \\
        runs/20260528-055134_phaseB/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# v5 #9 fix: dummy range E300MM 以外で Ericsson strict pattern (^E[39]\d{2}MM\d{6}$)
# に部分一致する文字列を「memorize 痕跡」として扱う。
# Why anchor なし: greedy_decode の出力は時として truncated になり (例: "E326MM11974"
# は 6 桁不足の partial)、strict regex に完全一致しなくても prefix が real 系であれば
# 漏洩リスク。`E[39]\d{2}MM\d+` で部分検出する。
_FORBIDDEN_PREFIX = re.compile(r"E[39]\d{2}MM\d+")


def is_memorized(text: str) -> bool:
    """text が「dummy 範囲外の Ericsson-like prefix」を含むか。

    >>> is_memorized("E300MM000001")
    False
    >>> is_memorized("E305MM999999")
    True
    >>> is_memorized("E326MM11974")  # partial 出力でも検知
    True
    >>> is_memorized("")
    False
    >>> is_memorized("GARBAGE")
    False
    """
    if not text:
        return False
    m = _FORBIDDEN_PREFIX.search(text)
    if not m:
        return False
    # E300MM プレフィクスは dummy として常に許可
    return not m.group(0).startswith("E300MM")


def check_summary(summary_path: Path) -> tuple[bool, list[dict]]:
    """summary.json を読んで memorize 痕跡を検出。

    Returns:
        (is_clean, leaks): clean=True なら leaks=[]、False なら問題のあるエントリ。
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    data = json.loads(summary_path.read_text())
    leaks: list[dict] = []
    for key in ("test_samples", "val_samples"):
        samples = data.get(key) or []
        for s in samples:
            pred = s.get("pred", "") if isinstance(s, dict) else ""
            if is_memorized(pred):
                leaks.append({"source": key, **(s if isinstance(s, dict) else {})})
    return (len(leaks) == 0, leaks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check training summary for memorized real-pattern serials.",
    )
    parser.add_argument("summary_path", type=Path,
                        help="Path to runs/<run>/summary.json")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress success message; only report leaks")
    args = parser.parse_args(argv)

    try:
        is_clean, leaks = check_summary(args.summary_path)
    except FileNotFoundError as e:
        print(f"[memorization-gate] ERROR: {e}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[memorization-gate] ERROR: failed to parse {args.summary_path}: {e}",
              file=sys.stderr)
        return 2

    if is_clean:
        if not args.quiet:
            print(f"[memorization-gate] OK: {args.summary_path} has no memorized prefixes",
                  file=sys.stderr)
        return 0

    print(
        f"[memorization-gate] BLOCKED: {args.summary_path} contains "
        f"{len(leaks)} prediction(s) with non-dummy Ericsson prefix:",
        file=sys.stderr,
    )
    for leak in leaks:
        print(
            f"    [{leak.get('source')}] pred={leak.get('pred')!r}  "
            f"gt={leak.get('gt')!r}  category={leak.get('category')!r}",
            file=sys.stderr,
        )
    print(
        "\nThis checkpoint must NOT be exported / published. Either:\n"
        "  1. Retrain with synth data regenerated using updated\n"
        "     `generate_random_ericsson_serial` (E300MM only).\n"
        "  2. If preds are obviously hallucinated (CER > ~30%), retrain anyway:\n"
        "     the model has not converged on the dummy distribution.\n"
        "See SECURITY.md and CLAUDE.md (dummy range policy).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
