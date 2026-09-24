import pytest

from career_advisor.graph import query as q
from career_advisor.rag.linking import link_entities
from career_advisor.rag.subgraph import fmt_number, retrieve_facts


def test_fmt_number_uses_vietnamese_decimal_comma():
    assert fmt_number(12.5) == "12,5" and fmt_number(12.0) == "12" and fmt_number(0.509, pct=True) == "50,9%"


def _facts(G, text):
    return retrieve_facts(G, link_entities(G, text))


def test_salary_fact_matches_graph_query_and_cites_postings(small_graph):
    facts = _facts(small_graph, "Lương kế toán tổng hợp ở Bắc Ninh bao nhiêu?")
    salary = next(f for f in facts if f.kind == "salary")

    summary = q.salary_summary(small_graph, q.title_id("kế toán tổng hợp"), q.province_id("bắc ninh"))
    assert summary.median in salary.values and summary.n in salary.values
    assert set(salary.sources) <= set(summary.posting_ids) | set(
        q.matching_postings(small_graph, q.title_id("kế toán tổng hợp"), q.province_id("bắc ninh"))
    )
    assert "12,5" in salary.text


def test_missing_skills_exclude_what_the_user_has(small_graph):
    facts = _facts(small_graph, "Tôi biết Excel, muốn làm kế toán tổng hợp ở Bắc Ninh")
    missing = next(f for f in facts if f.kind == "missing_skills")

    assert "excel" not in missing.text and "phần mềm kế toán" in missing.text
    assert missing.sources


def test_too_few_postings_gives_insufficient_fact_without_salary(small_graph):
    facts = _facts(small_graph, "Lương kế toán tổng hợp ở Hà Nội bao nhiêu?")  # chỉ 3 tin

    kinds = {f.kind for f in facts}
    assert "salary" not in kinds and "insufficient" in kinds


def test_every_fact_with_numbers_has_sources(small_graph):
    facts = _facts(
        small_graph, "Tôi biết Excel và MISA, muốn làm kế toán tổng hợp ở Bắc Ninh với 1–2 năm kinh nghiệm"
    )

    assert facts
    for fact in facts:
        if fact.kind != "insufficient":
            assert fact.values and fact.sources, fact


def test_nothing_linked_gives_no_facts(small_graph):
    assert _facts(small_graph, "Thời tiết hôm nay thế nào?") == []


@pytest.mark.parametrize(
    "text", ["Lương kế toán ở Nhật Bản là bao nhiêu?", "Lương kế toán tổng hợp năm 2023?"]
)
def test_out_of_scope_question_gives_out_of_scope_fact(small_graph, text):
    facts = _facts(small_graph, text)

    assert [f.kind for f in facts] == ["out_of_scope"]


def test_no_missing_skills_fact_when_the_user_names_no_skill(small_graph):
    facts = _facts(small_graph, "Lương trung vị của kế toán tổng hợp ở Bắc Ninh là bao nhiêu?")

    assert "missing_skills" not in {f.kind for f in facts}


def test_skill_lists_mark_soft_skills(small_graph):
    facts = _facts(small_graph, "Kỹ năng của kế toán tổng hợp là gì?")
    top = next(f for f in facts if f.kind == "top_skills")

    assert "cẩn thận (" in top.text and "kỹ năng mềm" in top.text.split("cẩn thận (")[1].split(")")[0]
    assert "kỹ năng mềm" not in top.text.split("excel (")[1].split(")")[0]
