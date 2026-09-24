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
- "đúng": khớp với tin gốc theo các quy ước bên dưới.
- "sai": có chỗ trái với tin gốc, bịa thêm, hoặc bỏ sót điều chính mà tin gốc ghi rõ.
- "không rõ": tin gốc không đủ thông tin để kiểm.

Quy ước của dữ liệu (làm theo quy ước thì là đúng, không phải lỗi):
- Lương: phần đầu tin ghi "0.0 triệu" nghĩa là tin không ghi lương, nên trích thành "thoả thuận" là đúng.
- Kinh nghiệm: quy ra số tháng rồi xếp nhóm: không yêu cầu (0 tháng), dưới 1 năm (1–11 tháng), 1–2 năm
  (12–24 tháng), 3–4 năm (25–48 tháng), từ 5 năm (trên 48 tháng). Ví dụ "1 năm" → "12 tháng (1–2 năm kinh
  nghiệm)" là đúng. Tin ghi một khoảng ("2-3 năm") thì lấy số nhỏ.
- Bằng cấp: lấy bậc thấp nhất tin chấp nhận ("Cao đẳng trở lên" → cao đẳng). Không xét chuyên ngành.
- Kỹ năng: tên được viết thường và gộp tên đồng nghĩa ("MS Excel → excel"), kỹ năng mềm vẫn giữ. Chỉ sai khi
  có kỹ năng không có trong tin, gộp sang tên mang nghĩa khác, hoặc bỏ sót kỹ năng chính tin ghi rõ.
Nếu có trường "sai", ghi lý do ngắn vào "ly_do".
Trả về JSON: {"skills": "...", "salary": "...", "experience": "...", "education": "...", "ly_do": "..."}"""


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


def parse_reason(text: str) -> str:
    try:
        data = json.loads(extract_json(text))
    except (ValueError, TypeError):
        return ""
    return str(data.get("ly_do", "")).strip() if isinstance(data, dict) else ""


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
