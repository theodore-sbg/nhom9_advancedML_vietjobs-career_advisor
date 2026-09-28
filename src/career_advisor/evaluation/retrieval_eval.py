"""Đánh giá truy xuất CV → tin: gộp danh sách để chấm (pooling) và tính chỉ số theo từng hệ."""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_advisor.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

K = 10


def pool(rankings: dict[str, dict[str, list[int]]]) -> dict[str, list[int]]:
    """Mỗi CV: hợp các tin mà ít nhất một hệ xếp vào top-k. Chỉ các tin này được chấm."""
    return {
        cv: sorted({pid for ranked in systems.values() for pid in ranked}) for cv, systems in rankings.items()
    }


def score_systems(
    rankings: dict[str, dict[str, list[int]]], grades: dict[str, dict[int, int]], k: int = K
) -> pd.DataFrame:
    """Trung bình Recall@k, MRR, nDCG@k trên các CV. CV không có tin phù hợp bị bỏ khỏi recall, nDCG."""
    systems = sorted({s for per_cv in rankings.values() for s in per_cv})
    rows = []
    for system in systems:
        recalls, mrrs, ndcgs = [], [], []
        for cv, per_cv in rankings.items():
            ranked, g = per_cv[system][:k], {pid: v for pid, v in grades.get(cv, {}).items() if v is not None}
            recalls.append(recall_at_k(ranked, g, k))
            mrrs.append(mrr(ranked, g))
            ndcgs.append(ndcg_at_k(ranked, g, k))

        def mean(values: list) -> float | None:
            kept = [v for v in values if v is not None]
            return float(np.mean(kept)) if kept else None

        rows.append(
            {
                "system": system,
                "recall@k": mean(recalls),
                "mrr": mean(mrrs),
                "ndcg@k": mean(ndcgs),
                "n_cv": len(rankings),
            }
        )
    return pd.DataFrame(rows)


def labeled_precision(
    rankings: dict[str, dict[str, list[int]]], labels: dict[str, dict[int, int]], k: int = K
) -> pd.DataFrame:
    """Kiểm chéo bằng nhãn tay: trong top-k của mỗi cách, xét các cặp đã có nhãn.

    Nhãn tay chỉ phủ một mẫu nhỏ, nên không tính được recall; chỉ đo tỷ lệ phù hợp của các cặp có nhãn.
    """
    systems = sorted({s for per_cv in rankings.values() for s in per_cv})
    rows = []
    for system in systems:
        grades = [
            labels[cv][pid]
            for cv, per_cv in rankings.items()
            for pid in per_cv[system][:k]
            if pid in labels.get(cv, {})
        ]
        n = len(grades)
        rows.append(
            {
                "system": system,
                "n_labeled": n,
                "mean_grade": float(np.mean(grades)) if n else None,
                "share_relevant": sum(g >= 2 for g in grades) / n if n else None,
                "share_partly": sum(g >= 1 for g in grades) / n if n else None,
            }
        )
    return pd.DataFrame(rows)


def paired_ndcg(
    rankings: dict[str, dict[str, list[int]]],
    grades: dict[str, dict[int, int]],
    system_a: str,
    system_b: str,
    k: int = K,
) -> tuple[list[float], list[float]]:
    """nDCG@k theo từng CV của hai hệ, ghép cặp. Bỏ CV mà một trong hai hệ không có nDCG."""
    a, b = [], []
    for cv, per_cv in rankings.items():
        g = {pid: v for pid, v in grades.get(cv, {}).items() if v is not None}
        x, y = ndcg_at_k(per_cv[system_a][:k], g, k), ndcg_at_k(per_cv[system_b][:k], g, k)
        if x is not None and y is not None:
            a.append(x)
            b.append(y)
    return a, b
