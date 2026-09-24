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
