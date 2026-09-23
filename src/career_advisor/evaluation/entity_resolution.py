"""Đánh giá bước gộp tên kỹ năng trên bộ nhãn tay."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from career_advisor.cleaning.skills import canonical_skill, split_compound
from career_advisor.evaluation.metrics import precision_recall

MIN_PRECISION = 0.95


def labeled_pairs(sheet: pd.DataFrame) -> pd.DataFrame:
    """Bỏ cặp "skip", thêm cột `truth` (True = cùng kỹ năng)."""
    pairs = sheet[sheet["label"].isin(["same", "different"])].copy()
    pairs["truth"] = pairs["label"] == "same"
    return pairs.reset_index(drop=True)


def _rule_canonical(name: str) -> str:
    parts = split_compound(name)
    return canonical_skill(parts[0] if parts else name)


def predict_same(pairs: pd.DataFrame, merges: pd.DataFrame) -> pd.Series:
    """Hệ gộp tên (luật + bảng `merges`) có coi a và b là một kỹ năng không."""
    final = dict(zip(merges["skill"], merges["canonical"], strict=True))

    def resolve(name: str) -> str:
        rule = _rule_canonical(name)
        return final.get(rule, rule)

    return pd.Series([resolve(a) == resolve(b) for a, b in zip(pairs["a"], pairs["b"], strict=True)])


def choose_threshold(
    truth: pd.Series,
    predict: Callable[[float], pd.Series],
    candidates: Sequence[float],
    min_precision: float = MIN_PRECISION,
) -> tuple[float | None, pd.DataFrame]:
    """Ngưỡng thấp nhất (recall cao nhất) vẫn đạt precision ≥ `min_precision`. Kèm bảng mọi ngưỡng."""
    rows = [
        {"threshold": t, **precision_recall(truth.tolist(), predict(t).tolist())} for t in sorted(candidates)
    ]
    table = pd.DataFrame(rows)
    ok = table[table["precision"].notna() & (table["precision"] >= min_precision)]
    return (float(ok["threshold"].iloc[0]) if len(ok) else None), table
