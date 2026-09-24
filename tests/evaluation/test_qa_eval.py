import pytest

from career_advisor.evaluation.qa_eval import aggregate, score_answer


def _q(kind, answer, group="one_hop", valid=(1, 2, 3)):
    return {"id": "q", "group": group, "type": kind, "answer": answer, "valid_posting_ids": list(valid)}


def _a(text, refuse=False, citations=()):
    return {"text": text, "refuse": refuse, "citations": list(citations)}


@pytest.mark.parametrize(
    "text, ok",
    [("Lương trung vị 14,5 triệu [#1]", True), ("khoảng 14.55 triệu", True), ("khoảng 15 triệu", False)],
)
def test_salary_answer_within_tolerance(text, ok):
    assert score_answer(_q("salary_title", {"median": 14.5, "n": 20}), _a(text))["correct"] is ok


def test_count_must_be_exact():
    q = _q("count_title_province", {"count": 24, "n": 24})

    assert score_answer(q, _a("Có 24 tin"))["correct"] and not score_answer(q, _a("Có 25 tin"))["correct"]


def test_skill_rate_compared_in_percent():
    q = _q("skill_rate", {"rate": 0.142, "n": 267})

    assert score_answer(q, _a("14,2% tin yêu cầu"))["correct"]
    assert not score_answer(q, _a("15% tin yêu cầu"))["correct"]


def test_named_answers_match_ignoring_case_and_tone_style():
    assert score_answer(_q("cooccur", {"skill": "tin học văn phòng"}), _a("Hay đi cùng Tin Học Văn Phòng"))[
        "correct"
    ]
    assert score_answer(_q("best_paid_title", {"title": "kế toán thuế"}), _a("Kế Toán Thuế cao nhất"))[
        "correct"
    ]


def test_gap_needs_salary_and_two_of_three_missing_skills():
    q = _q(
        "gap_and_salary", {"missing": ["misa", "excel", "word"], "salary_median": 12.0, "n": 30}, "multi_hop"
    )

    assert score_answer(q, _a("Thiếu MISA, Excel. Lương 12 triệu"))["correct"]
    assert not score_answer(q, _a("Thiếu MISA. Lương 12 triệu"))["correct"]
    assert not score_answer(q, _a("Thiếu MISA, Excel. Lương 13 triệu"))["correct"]


def test_refusal_scoring():
    out = _q("unknown_title", {"refuse": True}, "out_of_scope", valid=())
    answerable = _q("salary_title", {"median": 10.0, "n": 9})

    assert score_answer(out, _a("Không đủ dữ liệu", refuse=True))["correct"]
    assert not score_answer(out, _a("Lương 10 triệu"))["correct"]
    result = score_answer(answerable, _a("Không đủ dữ liệu", refuse=True))
    assert not result["correct"] and result["false_refusal"]


def test_citation_precision():
    q = _q("salary_title", {"median": 10.0, "n": 9}, valid=(1, 2))

    assert score_answer(q, _a("10 triệu", citations=(1, 2, 9)))["citation_precision"] == pytest.approx(2 / 3)
    assert score_answer(q, _a("10 triệu"))["citation_precision"] is None


def test_aggregate_by_system_and_group():
    rows = [
        {
            "system": "kg",
            "group": "one_hop",
            "correct": True,
            "citation_precision": 1.0,
            "false_refusal": False,
        },
        {
            "system": "kg",
            "group": "one_hop",
            "correct": False,
            "citation_precision": None,
            "false_refusal": True,
        },
        {
            "system": "kg",
            "group": "out_of_scope",
            "correct": True,
            "citation_precision": None,
            "false_refusal": False,
        },
    ]

    table = aggregate(rows).set_index(["system", "group"])

    assert table.loc[("kg", "one_hop"), "accuracy"] == 0.5
    assert table.loc[("kg", "one_hop"), "citation_precision"] == 1.0
    assert table.loc[("kg", "one_hop"), "false_refusal_rate"] == 0.5
    assert table.loc[("kg", "out_of_scope"), "accuracy"] == 1.0


def test_citation_grounding_checks_ids_against_what_the_llm_was_shown():
    q = _q("salary_title", {"median": 10.0, "n": 9}, valid=(1,))
    answer = {**_a("10 triệu", citations=(1, 2, 99)), "allowed_ids": [1, 2, 3]}

    assert score_answer(q, answer)["citation_grounded"] == pytest.approx(2 / 3)
