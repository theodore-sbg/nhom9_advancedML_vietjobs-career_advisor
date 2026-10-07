"""Chỉ số đánh giá dùng chung."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_recall(truth: Sequence[bool], pred: Sequence[bool]) -> dict:
    """Precision, recall, F1 cho bài toán nhị phân. Không có dự đoán dương thì precision là None."""
    tp = sum(t and p for t, p in zip(truth, pred, strict=True))
    fp = sum(p and not t for t, p in zip(truth, pred, strict=True))
    fn = sum(t and not p for t, p in zip(truth, pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(truth)}


# --- Truy xuất CV → tin. Điểm phù hợp: 0 không, 1 một phần, 2 phù hợp. ---
RELEVANT = 2


def recall_at_k(
    ranked: Sequence[int], grades: dict[int, int], k: int = 10, relevant: int = RELEVANT
) -> float | None:
    """Tỷ lệ tin phù hợp (trong tập đã chấm) có mặt ở top-k. Không có tin phù hợp nào thì None."""
    positives = {pid for pid, g in grades.items() if g >= relevant}
    if not positives:
        return None
    return len(positives & set(ranked[:k])) / len(positives)


def mrr(ranked: Sequence[int], grades: dict[int, int], relevant: int = RELEVANT) -> float:
    for rank, pid in enumerate(ranked, start=1):
        if grades.get(pid, 0) >= relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[int], grades: dict[int, int], k: int = 10) -> float | None:
    """nDCG với gain = điểm phù hợp (0/1/2). Tin chưa chấm coi là 0."""
    import math

    dcg = sum(grades.get(pid, 0) / math.log2(rank + 1) for rank, pid in enumerate(ranked[:k], start=1))
    ideal_gains = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
    return dcg / ideal if ideal > 0 else None


def cohen_kappa(a: Sequence[int], b: Sequence[int], weights: str | None = None) -> float:
    """Độ đồng thuận giữa hai người chấm (ở đây: LLM và người), đã trừ phần trùng do ngẫu nhiên."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, weights=weights))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 42) -> dict:
    """Bootstrap ghép cặp cho hiệu trung bình a − b (mỗi phần tử là điểm của cùng một CV ở hai hệ).

    Lấy mẫu lại các CV có hoàn lại. Khoảng tin cậy 95% theo phân vị. p hai phía = 2 × phần nhỏ hơn
    của các mẫu có hiệu ≤ 0 và ≥ 0, chặn trên ở 1.
    """
    if len(a) != len(b):
        raise ValueError(f"a và b phải ghép cặp: {len(a)} ≠ {len(b)}")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    # Rỗng, 1 cặp hay có NaN thì mọi mẫu bootstrap giống nhau hoặc là NaN, và p tính ra 0 — trông như
    # "có ý nghĩa". Báo lỗi thay vì trả một p sai.
    if len(diff) < 2:
        raise ValueError(f"cần ít nhất 2 cặp để bootstrap, có {len(diff)}")
    if np.isnan(diff).any():
        raise ValueError("điểm có NaN; bỏ các cặp thiếu điểm trước khi kiểm định")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return {
        "n": len(diff),
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_value": float(min(1.0, p)),
    }
