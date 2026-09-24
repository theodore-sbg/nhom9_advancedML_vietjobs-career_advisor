"""Chấm câu trả lời trên bộ câu hỏi (Task 11) và gộp kết quả theo hệ và nhóm câu."""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from career_advisor.cleaning.locations import tone_new_style
from career_advisor.rag.numbers import extract_numbers

SALARY_TOLERANCE = 0.1  # triệu đồng, hoặc 1% nếu lớn hơn
RATE_TOLERANCE = 0.2  # điểm phần trăm
MIN_MISSING_HITS = 2
SALARY_TYPES = ("salary_title", "salary_category_province", "salary_3way")


def _norm(text: str) -> str:
    return tone_new_style(unicodedata.normalize("NFC", text).lower())


def _has_number(text: str, target: float, tolerance: float) -> bool:
    return any(abs(x - target) <= tolerance for x in extract_numbers(text))


def _salary_ok(text: str, median: float) -> bool:
    return _has_number(text, median, max(SALARY_TOLERANCE, 0.01 * median))


def _is_correct(kind: str, expected: dict, text: str) -> bool:
    if kind in SALARY_TYPES:
        return _salary_ok(text, expected["median"])
    if kind == "count_title_province":
        return _has_number(text, expected["count"], 0)
    if kind == "skill_rate":
        return _has_number(text, expected["rate"] * 100, RATE_TOLERANCE)
    if kind == "cooccur":
        return _norm(expected["skill"]) in _norm(text)
    if kind == "best_paid_title":
        return _norm(expected["title"]) in _norm(text)
    if kind == "gap_and_salary":
        hits = sum(_norm(skill) in _norm(text) for skill in expected["missing"])
        return _salary_ok(text, expected["salary_median"]) and hits >= MIN_MISSING_HITS
    raise ValueError(f"Loại câu hỏi lạ: {kind}")


def score_answer(question: dict, answer: dict) -> dict:
    """`answer` có text, refuse, citations. Trả correct, false_refusal, citation_precision."""
    out_of_scope = question["group"] == "out_of_scope"
    if out_of_scope:
        correct = bool(answer["refuse"])
    elif answer["refuse"]:
        correct = False
    else:
        correct = _is_correct(question["type"], question["answer"], answer["text"])
    cited = set(answer["citations"])
    precision = (
        len(cited & set(question["valid_posting_ids"])) / len(cited) if cited and not out_of_scope else None
    )
    # Có căn cứ: mã tin được trích nằm trong những gì LLM đã được xem (không bịa mã tin).
    allowed = answer.get("allowed_ids")
    grounded = len(cited & set(allowed)) / len(cited) if cited and allowed is not None else None
    return {
        "correct": correct,
        "false_refusal": bool(answer["refuse"]) and not out_of_scope,
        "citation_precision": precision,
        "citation_grounded": grounded,
    }


def aggregate(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out = []
    for (system, group), g in frame.groupby(["system", "group"], sort=True):
        cited = g["citation_precision"].dropna()
        grounded = g["citation_grounded"].dropna() if "citation_grounded" in g else pd.Series(dtype=float)
        out.append(
            {
                "system": system,
                "group": group,
                "n": len(g),
                "accuracy": float(g["correct"].mean()),
                "citation_precision": float(cited.mean()) if len(cited) else np.nan,
                "citation_grounded": float(grounded.mean()) if len(grounded) else np.nan,
                "false_refusal_rate": float(g["false_refusal"].mean()),
            }
        )
    return pd.DataFrame(out)
