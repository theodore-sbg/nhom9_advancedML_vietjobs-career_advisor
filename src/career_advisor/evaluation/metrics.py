"""Chỉ số đánh giá dùng chung."""

from __future__ import annotations

from collections.abc import Sequence


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
