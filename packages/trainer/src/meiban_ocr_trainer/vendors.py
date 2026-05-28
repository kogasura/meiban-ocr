"""Vendor pattern 定義。runtime 側 `packages/runtime/src/vendors.ts` と同期。

HANDOFF.md §2 のベンダーパターンを Python 側で参照可能にする。
評価指標 (FPR, Acceptance Precision 等) は strict_regex をゲート条件として使う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VendorPattern:
    """ベンダー固有の serial 正規表現。

    Attributes:
        name: 識別子 (annotation の `vendor` フィールドと一致)
        strict_regex: 全体一致用 (^...$)
        partial_regex: 文字列内検索用 (anchor なし、誤検出許容)
    """

    name: str
    strict_regex: re.Pattern[str]
    partial_regex: re.Pattern[str]


ERICSSON = VendorPattern(
    name="ericsson",
    strict_regex=re.compile(r"^E[39]\d{2}MM\d{6}$"),
    partial_regex=re.compile(r"E[39]\d{2}MM\d{6}"),
)


VENDORS: dict[str, VendorPattern] = {
    ERICSSON.name: ERICSSON,
}


def get_vendor(name: str) -> VendorPattern:
    """名前から VendorPattern を取得。未登録なら ValueError。"""
    try:
        return VENDORS[name]
    except KeyError as e:
        raise ValueError(
            f"unknown vendor {name!r}. known: {list(VENDORS)}"
        ) from e


__all__ = ["VendorPattern", "ERICSSON", "VENDORS", "get_vendor"]
