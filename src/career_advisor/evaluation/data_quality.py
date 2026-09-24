"""Kiểm chất lượng dữ liệu: các trường đã trích (kỹ năng, lương, kinh nghiệm, bằng cấp) có khớp tin gốc không.

LLM chấm 200 tin; người kiểm tay 50 tin đầu của cùng mẫu để đo độ tin của LLM. Mỗi trường nhận một trong 3
nhận định: đúng, sai, không rõ (tin gốc không đủ để kiểm). Tỷ lệ sai chỉ tính trên đúng và sai.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from career_advisor.config import SEED
from career_advisor.evaluation.metrics import cohen_kappa
from career_advisor.llm import extract_json
from career_advisor.rag.subgraph import EXPERIENCE_TEXT, fmt_number

FIELDS = ("skills", "salary", "experience", "education")
FIELD_NAMES = {"skills": "Kỹ năng", "salary": "Lương", "experience": "Kinh nghiệm", "education": "Bằng cấp"}
VERDICTS = {"đúng": "correct", "sai": "wrong", "không rõ": "unclear"}
EDUCATION_TEXT = {
    "none": "không yêu cầu",
    "lower_secondary": "trung học cơ sở",
    "high_school": "trung học phổ thông",
    "vocational": "trung cấp",
    "college": "cao đẳng",
    "university": "đại học",
    "postgraduate": "sau đại học",
    "unspecified": "không ghi rõ",
}
MAX_SOURCE_CHARS = 2500

SYSTEM = "Bạn kiểm tra dữ liệu tin tuyển dụng. Chỉ trả JSON."
RULES = """Dưới đây là một tin tuyển dụng gốc và các trường đã được trích tự động từ tin đó.
Với từng trường, cho biết giá trị đã trích có đúng với tin gốc không:
- "đúng": khớp với tin gốc (được phép viết gọn hoặc chuẩn hoá, ví dụ "MS Excel → excel").
- "sai": có chỗ trái với tin gốc, bịa thêm, hoặc bỏ sót điều quan trọng tin gốc ghi rõ.
- "không rõ": tin gốc không đủ thông tin để kiểm.
Kỹ năng: sai nếu có kỹ năng không liên quan tới tin, hoặc bị gộp sai nghĩa. Lương "thoả thuận" là đúng nếu tin
không ghi mức lương cụ thể. Bằng cấp là bậc thấp nhất tin chấp nhận.
Trả về JSON: {"skills": "...", "salary": "...", "experience": "...", "education": "..."}"""


def sample_ids(index: pd.Index, n: int, seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.choice(np.asarray(index), size=n, replace=False)]


def _skills_text(skills: pd.DataFrame) -> str:
    parts = []
    for raw, skill in zip(skills["raw"], skills["skill"], strict=True):
        parts.append(skill if raw.strip().lower() == skill else f"{raw} → {skill}")
    return "; ".join(dict.fromkeys(parts)) or "(không có)"


def _salary_text(row: pd.Series) -> str:
    if row["salary_negotiable"] or pd.isna(row["salary_min"]):
        return "thoả thuận"
    low, high = fmt_number(row["salary_min"]), fmt_number(row["salary_max"])
    return f"{low} triệu" if low == high else f"{low}–{high} triệu"


def _experience_text(row: pd.Series) -> str:
    if pd.isna(row["experience_months"]):
        return "không ghi"
    level = EXPERIENCE_TEXT.get(row["experience_level"], row["experience_level"])
    return f"{int(row['experience_months'])} tháng ({level})"


def extracted_fields(row: pd.Series, skills: pd.DataFrame) -> dict[str, str]:
    """`row`: một dòng của postings.parquet. `skills`: các dòng của posting_skills.parquet của tin đó."""
    return {
        "skills": _skills_text(skills),
        "salary": _salary_text(row),
        "experience": _experience_text(row),
        "education": EDUCATION_TEXT.get(row["education_level"], row["education_level"]),
    }


def source_text(raw: pd.Series) -> str:
    """Tin gốc: phần đầu tin (chức danh, lương, kinh nghiệm, bằng cấp) và nội dung."""
    parts = [
        f"Chức danh: {raw['job_title']}",
        f"Lương: {raw['salary']}",
        f"Kinh nghiệm: {raw['experience_required']}",
        f"Bằng cấp: {raw['qualifications']}",
        f"Mô tả: {raw['description']}",
        f"Yêu cầu: {raw['requirements_text']}",
    ]
    return "\n".join(parts)[:MAX_SOURCE_CHARS]


def fields_block(fields: dict[str, str]) -> str:
    return "\n".join(f"- {f} ({FIELD_NAMES[f]}): {fields[f]}" for f in FIELDS)


def judge_prompt(source: str, fields: dict[str, str]) -> str:
    return f"{RULES}\n\nTIN GỐC:\n{source}\n\nCÁC TRƯỜNG ĐÃ TRÍCH:\n{fields_block(fields)}"


def parse_judgement(text: str) -> dict[str, str | None]:
    try:
        data = json.loads(extract_json(text))
    except (ValueError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {f: VERDICTS.get(str(data.get(f, "")).strip().lower()) for f in FIELDS}


def error_rates(labels: pd.DataFrame, fields=FIELDS) -> dict[str, dict]:
    out = {}
    for f in fields:
        clear = labels[f][labels[f].isin(["correct", "wrong"])]
        wrong = int((clear == "wrong").sum())
        out[f] = {"n": len(clear), "wrong": wrong, "error_rate": wrong / len(clear) if len(clear) else None}
    return out


def agreement(llm: pd.DataFrame, human: pd.DataFrame, fields=FIELDS) -> dict[str, dict]:
    """Độ khớp LLM–người trên các tin cả hai đều nói rõ đúng hay sai."""
    both = llm.merge(human, on="posting_id", suffixes=("_llm", "_human"))
    out = {}
    for f in fields:
        a, b = both[f"{f}_llm"], both[f"{f}_human"]
        keep = a.isin(["correct", "wrong"]) & b.isin(["correct", "wrong"])
        x, y = (a[keep] == "wrong").astype(int).tolist(), (b[keep] == "wrong").astype(int).tolist()
        out[f] = {
            "n": len(x),
            "agreement": float(np.mean(np.array(x) == np.array(y))) if x else None,
            "kappa": float(cohen_kappa(x, y)) if len(set(x) | set(y)) > 1 else None,
        }
    return out
