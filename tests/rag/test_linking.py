import pytest

from career_advisor.rag.linking import link_entities


def test_links_skills_title_and_province(small_graph):
    linked = link_entities(small_graph, "Tôi biết Excel và MISA, muốn làm kế toán tổng hợp ở Bắc Ninh")

    assert linked.skills == ["excel", "misa"]  # "kế toán tổng hợp" dùng cho chức danh, không tính là kỹ năng
    assert linked.titles == ["kế toán tổng hợp"]
    assert linked.provinces == ["bắc ninh"]


def test_title_with_generic_prefix_and_district(small_graph):
    linked = link_entities(small_graph, "Lương Nhân Viên Kinh Doanh ở Hà Đông bao nhiêu?")

    assert linked.titles == ["kinh doanh"]
    assert linked.provinces == ["hà nội"]


@pytest.mark.parametrize(
    "text, level",
    [
        ("với 1–2 năm kinh nghiệm", "1_2y"),
        ("tôi có 3 năm kinh nghiệm", "3_4y"),
        ("không yêu cầu kinh nghiệm", "none"),
        ("dưới 1 năm kinh nghiệm", "under_1y"),
        ("từ 5 năm kinh nghiệm", "5y_plus"),
    ],
)
def test_experience_level(small_graph, text, level):
    assert link_entities(small_graph, f"Làm kế toán tổng hợp {text}").experience_levels == [level]


def test_category_phrase(small_graph):
    linked = link_entities(small_graph, "Nhóm ngành tài chính kế toán có lương thế nào?")

    assert linked.categories == ["tài_chính_kế_toán"]


def test_out_of_scope_signals(small_graph):
    foreign = link_entities(small_graph, "Lương kế toán ở Nhật Bản là bao nhiêu?")
    old = link_entities(small_graph, "Lương kế toán tổng hợp năm 2023?")
    current = link_entities(small_graph, "Lương kế toán tổng hợp năm 2025?")

    assert foreign.foreign and foreign.provinces == []
    assert old.other_years == [2023]
    assert current.other_years == []


def test_unknown_title_links_nothing(small_graph):
    linked = link_entities(small_graph, "Lương trung vị của phi công ở Hà Nội là bao nhiêu?")

    assert linked.titles == [] and linked.provinces == ["hà nội"]


def test_question_words_are_not_read_as_skills(small_graph):
    linked = link_entities(small_graph, "Lương trung vị của nhóm ngành kinh doanh bán hàng là bao nhiêu?")

    assert linked.skills == []


def test_category_phrase_is_fully_consumed_before_titles(small_graph):
    linked = link_entities(small_graph, "Trong nhóm ngành công nghệ thông tin kỹ thuật số, lương ra sao?")

    assert linked.categories == ["công_nghệ_thông_tin_kỹ_thuật_số"]
    assert linked.titles == []
