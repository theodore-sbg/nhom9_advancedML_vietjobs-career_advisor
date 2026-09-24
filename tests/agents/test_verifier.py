import pytest

from career_advisor.agents.verifier import verify
from career_advisor.rag.linking import link_entities
from career_advisor.rag.subgraph import retrieve_facts

QUESTION = "Tôi biết Excel và 2 phần mềm khác. Lương kế toán tổng hợp ở Bắc Ninh bao nhiêu?"


@pytest.fixture
def facts(small_graph):
    return retrieve_facts(small_graph, link_entities(small_graph, QUESTION))


@pytest.fixture
def salary(facts):
    return next(f for f in facts if f.kind == "salary")


def _answer(salary, text):
    return text.format(median=salary.text.split(": ")[1].split(" triệu")[0], src=salary.sources[0])


def test_faithful_answer_passes(facts, salary):
    text = _answer(
        salary, "Lương trung vị là {median} triệu đồng/tháng [#{src}]. Bạn đã biết 2 phần mềm khác."
    )

    result = verify(text, [salary.sources[0]], facts, QUESTION)

    assert result.ok, result.reasons


def test_list_numbering_is_not_a_number(facts, salary):
    text = _answer(salary, "1. Lương trung vị {median} triệu [#{src}].\n2. Nên học thêm MISA.")

    assert verify(text, [], facts, QUESTION).ok


@pytest.mark.parametrize(
    "planted",
    [
        "Lương trung vị là 13 triệu.",  # số khác
        "Lương trung vị khoảng 15 triệu.",  # làm tròn khác đi
        "Lương trung vị là 12.600.000 đồng.",  # viết kiểu đồng nhưng sai
        "Lương trung vị là 12tr6.",  # viết tắt nhưng sai
        "51% tin yêu cầu MISA.",  # tỷ lệ sai
        "Có 99 tin tuyển.",  # số tin sai
        "Lương có thể lên tới 40 triệu.",  # số bịa thêm
    ],
)
def test_planted_wrong_number_is_blocked(facts, planted):
    result = verify(planted, [], facts, QUESTION)

    assert not result.ok and result.unsupported_numbers


def test_citation_outside_facts_is_blocked(facts, salary):
    text = _answer(salary, "Lương trung vị là {median} triệu [#999999].")

    result = verify(text, [], facts, QUESTION)

    assert not result.ok and result.unsupported_citations == [999999]


def test_citation_list_is_also_checked(facts, salary):
    result = verify("Có dữ liệu.", [salary.sources[0], 424242], facts, QUESTION)

    assert result.unsupported_citations == [424242]


def test_reasons_are_readable(facts):
    result = verify("Lương là 13 triệu [#999999].", [], facts, QUESTION)

    assert any("13" in r for r in result.reasons) and any("999999" in r for r in result.reasons)
