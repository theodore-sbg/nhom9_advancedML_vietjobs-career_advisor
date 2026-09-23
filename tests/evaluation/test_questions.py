import pandas as pd
import pytest

from career_advisor.evaluation import questions as qs


def _postings() -> pd.DataFrame:
    rows = []
    # 6 tin kế toán ở Hà Nội, 2 ở Bắc Ninh (ít hơn 5 tin), 5 tin lập trình ở TP.HCM, 1 tin lương thoả thuận.
    for salary in (10, 11, 12, 13, 14, 15):
        rows.append(("acc", ["hà nội"], salary, "1_2y", "kế toán", "Kế Toán"))
    for salary in (9, 20):
        rows.append(("acc", ["bắc ninh"], salary, "1_2y", "kế toán", "Kế Toán"))
    for salary in (20, 25, 30, 35, 40):
        rows.append(("it", ["hồ chí minh"], salary, "3_4y", "lập trình viên", "Lập Trình Viên"))
    rows.append(("it", ["hồ chí minh"], None, "3_4y", "lập trình viên", "Lập Trình Viên"))
    df = pd.DataFrame(
        rows,
        columns=[
            "category",
            "provinces",
            "salary_mid",
            "experience_level",
            "job_title_node",
            "job_title_display",
        ],
    )
    df.index.name = "posting_id"
    return df


def _skills() -> pd.DataFrame:
    rows = [(i, "excel") for i in range(8)] + [(i, "misa") for i in range(0, 8, 2)] + [(1, "word")]
    rows += [(i, "python") for i in range(8, 14)] + [(i, "git") for i in range(8, 12)]
    return pd.DataFrame(rows, columns=["posting_id", "skill"])


def test_salary_median_uses_only_postings_with_salary():
    ans = qs.salary_answer(_postings(), job_title_node="lập trình viên")

    assert ans["median"] == 30.0 and ans["n"] == 5
    assert set(ans["posting_ids"]) == {8, 9, 10, 11, 12}


def test_salary_answer_filters_on_province_membership():
    ans = qs.salary_answer(_postings(), category="acc", province="hà nội")

    assert ans["median"] == 12.5 and ans["n"] == 6


def test_skill_rate_matches_hand_count():
    ans = qs.skill_rate_answer(_postings(), _skills(), "kế toán", "misa")

    assert ans["rate"] == pytest.approx(4 / 8)
    assert ans["n_with"] == 4 and ans["n_group"] == 8


def test_top_cooccurring_skill():
    ans = qs.cooccur_answer(_skills(), "python")

    assert ans["skill"] == "git" and ans["count"] == 4


def test_gap_answer_excludes_owned_skills_and_ranks_by_rate():
    ans = qs.gap_answer(_postings(), _skills(), "kế toán", have=["excel"])

    assert ans["missing"][0] == "misa"
    assert "excel" not in ans["missing"]


def test_best_paid_title_in_category_needs_enough_postings():
    ans = qs.best_paid_title_answer(_postings(), "acc", min_postings=5)

    assert ans["title"] == "kế toán"


def test_build_question_set_has_all_groups_and_balanced_splits():
    postings, skills = _postings(), _skills()

    questions = qs.build_question_set(postings, skills, counts=qs.small_counts(), min_n=5, seed=0)

    groups = {q["group"] for q in questions}
    assert groups == {"one_hop", "multi_hop", "out_of_scope"}
    for group in groups:
        splits = [q["split"] for q in questions if q["group"] == group]
        assert abs(splits.count("dev") - splits.count("test")) <= 1
    assert len({q["id"] for q in questions}) == len(questions)


def test_answerable_questions_never_rest_on_fewer_than_min_n_postings():
    questions = qs.build_question_set(_postings(), _skills(), counts=qs.small_counts(), min_n=5, seed=0)

    for q in questions:
        if q["group"] != "out_of_scope" and "n" in q["answer"]:
            assert q["answer"]["n"] >= 5


def test_small_combination_becomes_a_refusal_question():
    questions = qs.build_question_set(_postings(), _skills(), counts=qs.small_counts(), min_n=5, seed=0)

    refusals = [q for q in questions if q["group"] == "out_of_scope"]
    assert all(q["answer"] == {"refuse": True} for q in refusals)
    assert any(q["type"] == "too_few_postings" for q in refusals)


def test_cooccur_answer_for_a_skill_that_never_appears_with_others():
    skills = pd.DataFrame({"posting_id": [0, 1], "skill": ["lẻ loi", "khác"]})

    ans = qs.cooccur_answer(skills, "lẻ loi")

    assert ans["skill"] is None and not ans["unique_top"]
