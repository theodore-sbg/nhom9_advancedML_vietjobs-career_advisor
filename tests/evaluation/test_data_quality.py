import json

import pandas as pd
import pytest

from career_advisor.evaluation.data_quality import (
    FIELDS,
    agreement,
    error_rates,
    extracted_fields,
    judge_prompt,
    parse_judgement,
    sample_ids,
)


def _posting(**overrides):
    row = {
        "salary_min": 10.0,
        "salary_max": 15.0,
        "salary_negotiable": False,
        "experience_months": 24,
        "experience_level": "1_2y",
        "education_level": "college",
    }
    return pd.Series({**row, **overrides})


def test_sample_ids_are_fixed_by_the_seed_and_unique():
    index = pd.Index(range(1000))

    first, second = sample_ids(index, 200, seed=42), sample_ids(index, 200, seed=42)

    assert first == second and len(set(first)) == 200


def test_extracted_fields_show_merges_and_readable_values():
    skills = pd.DataFrame({"raw": ["MS Excel", "Kế toán"], "skill": ["excel", "kế toán"]})

    fields = extracted_fields(_posting(), skills)

    assert fields["skills"] == "MS Excel → excel; kế toán"
    assert fields["salary"] == "10–15 triệu"
    assert fields["experience"] == "24 tháng (1–2 năm kinh nghiệm)"
    assert fields["education"] == "cao đẳng"


def test_negotiable_salary_and_missing_skills():
    fields = extracted_fields(_posting(salary_negotiable=True), pd.DataFrame(columns=["raw", "skill"]))

    assert fields["salary"] == "thoả thuận"
    assert fields["skills"] == "(không có)"


def test_judge_prompt_lists_every_field():
    prompt = judge_prompt("TIN GỐC", {f: f"giá trị {f}" for f in FIELDS})

    assert "TIN GỐC" in prompt and all(f"giá trị {f}" in prompt for f in FIELDS)


def test_parse_judgement_reads_verdicts_and_ignores_unknown_words():
    text = json.dumps({"skills": "đúng", "salary": "sai", "experience": "không rõ", "education": "ổn"})

    assert parse_judgement(text) == {
        "skills": "correct",
        "salary": "wrong",
        "experience": "unclear",
        "education": None,
    }


def test_parse_judgement_of_broken_json_is_all_none():
    assert parse_judgement("xin lỗi") == {f: None for f in FIELDS}


def test_error_rates_ignore_unclear_and_missing():
    labels = pd.DataFrame(
        {
            "skills": ["correct", "wrong", "unclear", None],
            "salary": ["correct", "correct", "correct", "wrong"],
            "experience": ["correct"] * 4,
            "education": ["wrong"] * 4,
        }
    )

    rates = error_rates(labels)

    assert rates["skills"] == {"n": 2, "wrong": 1, "error_rate": 0.5}
    assert rates["salary"]["error_rate"] == pytest.approx(0.25)
    assert rates["education"]["error_rate"] == 1.0


def test_agreement_uses_postings_both_judged_with_a_clear_verdict():
    llm = pd.DataFrame({"posting_id": [1, 2, 3, 4], "skills": ["correct", "wrong", "correct", "unclear"]})
    human = pd.DataFrame({"posting_id": [1, 2, 3, 4], "skills": ["correct", "wrong", "wrong", "correct"]})

    result = agreement(llm, human, fields=("skills",))

    assert result["skills"]["n"] == 3
    assert result["skills"]["agreement"] == pytest.approx(2 / 3)
    assert -1 <= result["skills"]["kappa"] <= 1
