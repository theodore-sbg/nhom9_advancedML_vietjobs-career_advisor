from career_advisor.evaluation.agent_eval import dominant_category, mentioned_share
from career_advisor.graph import query as q
from career_advisor.rag.linking import LinkedEntities
from career_advisor.rag.subgraph import group_node


def test_mentioned_share_ignores_case_and_tone_style():
    text = "Nên học MISA trước, rồi Phần mềm kế toán."

    assert mentioned_share(text, ["misa", "phần mềm kế toán", "excel", "cẩn thận"]) == 0.5


def test_mentioned_share_of_no_skill_is_none():
    assert mentioned_share("gì cũng được", []) is None


def test_dominant_category_of_a_title_is_where_most_of_its_postings_are(small_graph):
    assert dominant_category(small_graph, q.title_id("kế toán tổng hợp")) == "tài_chính_kế_toán"


def test_dominant_category_of_a_category_is_itself(small_graph):
    node = q.category_id("kinh_doanh_bán_hàng")

    assert dominant_category(small_graph, node) == "kinh_doanh_bán_hàng"


def test_group_node_prefers_title_then_category():
    assert group_node(LinkedEntities(titles=["kế toán"], categories=["x"])) == q.title_id("kế toán")
    assert group_node(LinkedEntities(categories=["x"])) == q.category_id("x")
    assert group_node(LinkedEntities()) is None
