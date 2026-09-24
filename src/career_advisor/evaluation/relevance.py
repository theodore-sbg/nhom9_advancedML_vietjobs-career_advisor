"""LLM chấm độ phù hợp giữa CV và tin tuyển dụng (0/1/2), dùng làm nhãn cho đánh giá truy xuất.

Nhãn của LLM được kiểm lại bằng một mẫu nhãn tay (Cohen's κ), vì chấm bằng LLM có thể lệch.
"""

from __future__ import annotations

import json

import pandas as pd

from career_advisor.llm import LLMGenerationError, extract_json

BATCH = 8
MAX_REQUIREMENTS_CHARS = 400
MAX_SKILLS = 12

SYSTEM = (
    "Bạn là chuyên viên tuyển dụng ở Việt Nam, đánh giá mức phù hợp giữa CV và tin tuyển dụng. Chỉ trả JSON."
)
RULES = """Chấm từng tin tuyển dụng theo mức phù hợp với CV dưới đây:
- 2: phù hợp — đúng nghề hoặc lĩnh vực của ứng viên, và CV đáp ứng phần lớn yêu cầu chính.
- 1: phù hợp một phần — cùng lĩnh vực nhưng lệch vai trò hoặc cấp độ, hoặc thiếu nhiều kỹ năng chính.
- 0: không phù hợp.
Không xét giới tính, tuổi hay mức lương.

Trả về JSON dạng {"grades": [{"id": 1, "grade": 2}, {"id": 2, "grade": 0}, ...]}.
"""


def posting_summary(row: pd.Series, skills: list[str]) -> str:
    provinces = ", ".join(p.title() for p in row["provinces"])
    requirements = str(row.get("requirements_text") or "")[:MAX_REQUIREMENTS_CHARS]
    return (
        f"{row['job_title']} | ngành: {str(row['category']).replace('_', ' ')} | nơi làm: {provinces} | "
        f"kỹ năng: {', '.join(skills[:MAX_SKILLS])} | yêu cầu: {requirements}"
    )


def relevance_prompt(cv: str, summaries: list[str]) -> str:
    listing = "\n".join(f"[{k}] {s}" for k, s in enumerate(summaries, start=1))
    return f"{RULES}\nCV:\n{cv}\n\nTin tuyển dụng:\n{listing}"


def read_grades(text: str, n: int) -> tuple[list[int | None], bool]:
    grades: list[int | None] = [None] * n
    try:
        data = json.loads(extract_json(text))
        for item in data["grades"] if isinstance(data, dict) else data:
            k, grade = int(item["id"]), int(item["grade"])
            if 1 <= k <= n and grade in (0, 1, 2):
                grades[k - 1] = grade
    except (ValueError, KeyError, TypeError):
        return [None] * n, False
    return grades, all(g is not None for g in grades)


def _judge_batch(client, cv: str, items: list[tuple[int, str]]) -> list[int | None]:
    try:
        text = client.complete(relevance_prompt(cv, [s for _, s in items]), system=SYSTEM, json_output=True)
    except LLMGenerationError:
        if len(items) == 1:
            return [None]
        half = len(items) // 2
        return _judge_batch(client, cv, items[:half]) + _judge_batch(client, cv, items[half:])
    return read_grades(text, len(items))[0]


def judge_relevance(
    client, cv: str, items: list[tuple[int, str]], batch_size: int = BATCH
) -> dict[int, int | None]:
    """Chấm các tin `items` = [(posting_id, tóm tắt)]. Tin chấm không được thì None."""
    grades: dict[int, int | None] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        grades.update(zip([pid for pid, _ in batch], _judge_batch(client, cv, batch), strict=True))
    return grades


def sample_for_human(llm_grades: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Lấy `n` cặp để người gán tay, chia đều theo điểm LLM (0/1/2) để κ đo được ở cả 3 mức.

    Mức nào không đủ cặp thì lấy hết, phần thiếu bù ngẫu nhiên từ các cặp còn lại.
    Tệp nhãn không chứa điểm LLM, để người gán không bị ảnh hưởng.
    """
    graded = llm_grades.dropna(subset=["grade"]).reset_index(drop=True)
    quota = n // 3
    picked = pd.concat(
        [g.sample(min(quota, len(g)), random_state=seed) for _, g in graded.groupby("grade", sort=True)]
    )
    rest = graded.drop(picked.index)
    picked = pd.concat([picked, rest.sample(min(n - len(picked), len(rest)), random_state=seed)])
    picked = picked.sample(frac=1, random_state=seed)  # xáo thứ tự để không lộ điểm LLM qua vị trí
    return picked[["cv_id", "posting_id"]].assign(label="").reset_index(drop=True)
