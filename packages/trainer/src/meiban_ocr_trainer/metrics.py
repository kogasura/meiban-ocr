"""評価指標。positive/negative の category 別に集計し、reject 性能を可視化する。

設計判断:
- **判定ゲートは pattern_match (strict regex)**。runtime も pattern を最終ゲートに
  使う想定なので、訓練評価でも同条件で測る。
- 空出力 (empty) は pattern_match=False の特殊ケースとして扱う。FPR は両側
  (nonempty / pattern) で別途算出し、「空でないが pattern を満たさない」ケースも
  可視化する。
- subkind 別 breakdown を出すと、どの種類の negative に弱いか (背景 / other_text /
  mined 等) が見える。Phase 1.5 のハードネガティブ追補に直結する情報。

用語:
- accepted:        model 出力が strict_regex に合致 (= システムが「コード」と判定)
- rejected:        空出力 or 不正パターン (= システムが「非コード」と判定)
- FPR_pattern:     negative のうち accepted の割合 (誤受容率)
- Rejection Recall: 1 - FPR_pattern (正しく拒否した割合)
- Acceptance Precision: accepted のうち GT と一致した割合 (受容時の正確性)
- Acceptance Recall:    positive のうち accepted の割合 (取りこぼしの少なさ)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field


def _safe_div(num: float | int, den: float | int) -> float:
    return float(num / den) if den else 0.0


def _cer(pred: str, gt: str) -> float:
    """1サンプルの編集距離 / 参照長。Levenshtein, 文字単位。"""
    if not gt:
        return 1.0 if pred else 0.0
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n] / n


@dataclass
class PerSubkindStat:
    """negative の subkind 1種類分の集計。"""

    n: int = 0
    n_empty: int = 0
    n_pattern_match: int = 0

    @property
    def fpr_pattern(self) -> float:
        return _safe_div(self.n_pattern_match, self.n)

    @property
    def fpr_nonempty(self) -> float:
        return _safe_div(self.n - self.n_empty, self.n)


@dataclass
class EvaluationReport:
    """1 split 分の評価結果。

    全 None フィールドは「該当サンプル無し」を示す (例: positive が無ければ cer=None)。
    """

    n_samples: int
    n_pos: int
    n_neg: int

    # positive 側指標
    cer: float | None = None
    em: float | None = None
    acceptance_recall: float | None = None
    em_among_accepted: float | None = None  # = Acceptance Precision

    # negative 側指標 (reject 性能)
    fpr_nonempty: float | None = None
    fpr_pattern: float | None = None
    rejection_recall: float | None = None

    # subkind 別 negative breakdown
    per_subkind: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    predictions: list[str],
    ground_truths: list[str],
    categories: list[str],
    subkinds: list[str] | None = None,
    pattern: re.Pattern[str] | None = None,
    confidences: list[float] | None = None,
    confidence_threshold: float | None = None,
) -> EvaluationReport:
    """予測リストから per-category 指標を計算。

    Args:
        predictions: モデル出力文字列のリスト
        ground_truths: 正解 (negative は "")
        categories: 各サンプルの category ('positive' | 'negative')
        subkinds: 各サンプルの subkind (negative のみ意味あり)
        pattern: accept/reject ゲートに使う vendor の strict regex。
                 None の場合は acceptance ベース指標が None になる。
        confidences: 各サンプルのモデル出力 confidence (0-1)。tokenizer の
                     `greedy_decode_with_conf` から取得した値を想定。
        confidence_threshold: confidence ゲートの閾値。`confidences` と両方指定された
                              場合、accept = pattern_match AND (confidence >= threshold)
                              となる。どちらかが None なら confidence ゲートは無効化。
    """
    n = len(predictions)
    if not (n == len(ground_truths) == len(categories)):
        raise ValueError(
            f"length mismatch: preds={n}, gts={len(ground_truths)}, "
            f"cats={len(categories)}"
        )
    if subkinds is None:
        subkinds = [""] * n
    elif len(subkinds) != n:
        raise ValueError(f"subkinds length {len(subkinds)} != preds {n}")
    if confidences is not None and len(confidences) != n:
        raise ValueError(f"confidences length {len(confidences)} != preds {n}")
    for c in categories:
        if c not in ("positive", "negative"):
            raise ValueError(f"unknown category: {c!r}")

    use_conf_gate = (
        confidences is not None and confidence_threshold is not None
    )

    def is_pattern_match(s: str) -> bool:
        # 単純な pattern match (confidence は見ない)。
        # FPR_pattern と RejectionRecall は pattern のみで定義。
        return bool(pattern.match(s)) if (pattern is not None and s) else False

    def is_accepted(idx: int) -> bool:
        """accept ゲート: pattern match (+ optional confidence threshold)。"""
        if not is_pattern_match(predictions[idx]):
            return False
        if use_conf_gate and confidences[idx] < confidence_threshold:
            return False
        return True

    # positive 側集計
    pos_indices = [i for i, c in enumerate(categories) if c == "positive"]
    n_pos = len(pos_indices)
    cer: float | None = None
    em: float | None = None
    acceptance_recall: float | None = None
    em_among_accepted: float | None = None
    if n_pos > 0:
        pos_cers = [_cer(predictions[i], ground_truths[i]) for i in pos_indices]
        cer = sum(pos_cers) / n_pos
        em = _safe_div(
            sum(1 for i in pos_indices if predictions[i] == ground_truths[i]),
            n_pos,
        )
        if pattern is not None:
            pos_accepted = [i for i in pos_indices if is_accepted(i)]
            n_acc = len(pos_accepted)
            acceptance_recall = _safe_div(n_acc, n_pos)
            em_among_accepted = (
                _safe_div(
                    sum(1 for i in pos_accepted if predictions[i] == ground_truths[i]),
                    n_acc,
                )
                if n_acc > 0
                else None
            )

    # negative 側集計
    neg_indices = [i for i, c in enumerate(categories) if c == "negative"]
    n_neg = len(neg_indices)
    fpr_nonempty: float | None = None
    fpr_pattern: float | None = None
    rejection_recall: float | None = None
    per_subkind: dict[str, dict[str, float]] = {}
    if n_neg > 0:
        n_neg_nonempty = sum(1 for i in neg_indices if predictions[i])
        fpr_nonempty = _safe_div(n_neg_nonempty, n_neg)
        if pattern is not None:
            # FPR_pattern は **accept ゲート** (pattern + 任意で confidence) で算出。
            # confidence threshold を入れると FPR が下がる方向に動くので、
            # gate を変えた効果がそのままここに反映される。
            n_neg_accepted = sum(1 for i in neg_indices if is_accepted(i))
            fpr_pattern = _safe_div(n_neg_accepted, n_neg)
            rejection_recall = 1.0 - fpr_pattern

        bucket: dict[str, PerSubkindStat] = defaultdict(PerSubkindStat)
        for i in neg_indices:
            key = subkinds[i] or "unspecified"
            stat = bucket[key]
            stat.n += 1
            if not predictions[i]:
                stat.n_empty += 1
            if is_accepted(i):
                stat.n_pattern_match += 1
        per_subkind = {
            k: {
                "n": s.n,
                "fpr_pattern": s.fpr_pattern,
                "fpr_nonempty": s.fpr_nonempty,
            }
            for k, s in sorted(bucket.items())
        }

    return EvaluationReport(
        n_samples=n,
        n_pos=n_pos,
        n_neg=n_neg,
        cer=cer,
        em=em,
        acceptance_recall=acceptance_recall,
        em_among_accepted=em_among_accepted,
        fpr_nonempty=fpr_nonempty,
        fpr_pattern=fpr_pattern,
        rejection_recall=rejection_recall,
        per_subkind=per_subkind,
    )


def format_report(report: EvaluationReport, label: str = "") -> str:
    """ログ出力用の複数行サマリ。None は N/A 表示。"""
    def _fmt(v: float | None, spec: str = ".4f") -> str:
        return f"{v:{spec}}" if v is not None else "  N/A "

    prefix = f"[{label}] " if label else ""
    head = f"{prefix}n={report.n_samples} (pos={report.n_pos}, neg={report.n_neg})"
    pos_line = (
        f"  pos: CER={_fmt(report.cer)}  EM={_fmt(report.em, '.3f')}  "
        f"AcceptRecall={_fmt(report.acceptance_recall, '.3f')}  "
        f"AcceptPrecision(EM|accepted)={_fmt(report.em_among_accepted, '.3f')}"
    )
    neg_line = (
        f"  neg: FPR_pattern={_fmt(report.fpr_pattern, '.3f')}  "
        f"FPR_nonempty={_fmt(report.fpr_nonempty, '.3f')}  "
        f"RejectRecall={_fmt(report.rejection_recall, '.3f')}"
    )
    parts = [head, pos_line, neg_line]
    if report.per_subkind:
        parts.append("  neg by subkind:")
        for k, v in report.per_subkind.items():
            parts.append(
                f"    - {k}: n={int(v['n'])}  fpr_pattern={v['fpr_pattern']:.3f}  "
                f"fpr_nonempty={v['fpr_nonempty']:.3f}"
            )
    return "\n".join(parts)


__all__ = ["EvaluationReport", "compute_metrics", "format_report"]
