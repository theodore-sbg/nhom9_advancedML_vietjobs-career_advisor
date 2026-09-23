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
