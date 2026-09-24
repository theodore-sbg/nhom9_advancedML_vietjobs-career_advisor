import json
import re

import pandas as pd

from career_advisor.evaluation.relevance import judge_relevance, posting_summary, read_grades
from career_advisor.llm import FakeLLMClient, LLMGenerationError


def test_read_grades_parses_and_flags_missing():
    text = '{"grades": [{"id": 1, "grade": 2}, {"id": 3, "grade": 0}]}'

    assert read_grades(text, 3) == ([2, None, 0], False)
    assert read_grades("hỏng", 2) == ([None, None], False)


def test_read_grades_rejects_out_of_range_grade():
    grades, complete = read_grades('{"grades": [{"id": 1, "grade": 5}]}', 1)

    assert grades == [None] and not complete


def test_posting_summary_is_short_and_keeps_key_fields():
    row = pd.Series(
        {
            "job_title": "Kế toán",
            "category": "tài_chính",
            "provinces": ["hà nội"],
            "requirements_text": "x" * 2000,
        }
    )

    text = posting_summary(row, ["misa", "excel"])

    assert "Kế toán" in text and "misa, excel" in text
    assert len(text) < 700


def test_location_is_neither_shown_nor_judged():
    from career_advisor.evaluation.relevance import RULES

    row = pd.Series(
        {
            "job_title": "Kế toán",
            "category": "tài_chính",
            "provinces": ["hà nội"],
            "requirements_text": "MISA",
        }
    )

    # Độ phù hợp chỉ xét nghề nghiệp; nơi làm do bộ lọc tỉnh trên giao diện xử lý.
    assert "Hà Nội" not in posting_summary(row, ["misa"])
    assert "Không xét nơi làm" in RULES


def _grader(grade_of):
    def respond(prompt: str) -> str:
        ids = re.findall(r"^\[(\d+)\] (.+)$", prompt, flags=re.M)
        return json.dumps({"grades": [{"id": int(k), "grade": grade_of(t)} for k, t in ids]})

    return respond


def test_judge_relevance_maps_grades_back_to_posting_ids():
    client = FakeLLMClient(_grader(lambda t: 2 if "Kế toán" in t else 0))
    items = [(10, "Kế toán tổng hợp"), (20, "Lập trình viên"), (30, "Kế toán kho")]

    grades = judge_relevance(client, "CV kế toán", items, batch_size=2)

    assert grades == {10: 2, 20: 0, 30: 2}
    assert client.calls == 2


def test_judge_relevance_splits_batch_that_fails_to_generate():
    good = _grader(lambda t: 1)

    def respond(prompt: str) -> str:
        if "Hỏng" in prompt:
            raise LLMGenerationError("repeat limit")
        return good(prompt)

    items = [(1, "Tốt A"), (2, "Hỏng"), (3, "Tốt B")]

    grades = judge_relevance(FakeLLMClient(respond), "CV", items, batch_size=3)

    assert grades == {1: 1, 2: None, 3: 1}


def test_human_sample_is_balanced_across_llm_grades_and_hides_the_llm_grade():
    from career_advisor.evaluation.relevance import sample_for_human

    llm = pd.DataFrame(
        {
            "cv_id": [f"cv{i % 5}" for i in range(90)],
            "posting_id": range(90),
            "grade": [0] * 60 + [1] * 20 + [2] * 9 + [None],
        }
    )

    sample = sample_for_human(llm, n=30, seed=0)

    assert len(sample) == 30
    assert list(sample.columns) == ["cv_id", "posting_id", "label"]
    picked = llm.set_index("posting_id").loc[sample.posting_id, "grade"]
    counts = picked.value_counts()
    # Mỗi mức lấy tối đa 10; mức 2 chỉ có 9 nên lấy hết, cặp còn thiếu bù từ mức khác.
    assert counts[2] == 9 and counts[0] >= 10 and counts[1] >= 10
    assert picked.notna().all() and sample.posting_id.is_unique
    assert (sample.label == "").all()
