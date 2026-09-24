import json

import pandas as pd

from career_advisor.app_support import answer_subgraph, match_table, skills_table, subgraph_html
from career_advisor.graph import query as q
from career_advisor.rag.answer import Answer
from career_advisor.rag.linking import LinkedEntities
from career_advisor.rag.subgraph import Fact, fmt_number


def test_skills_table_shows_rate_as_percent_and_chi2(small_graph):
    stats = q.missing_skills(small_graph, ["excel"], q.title_id("kế toán tổng hợp"), k=3)

    table = skills_table(stats)

    assert list(table.columns) == ["Kỹ năng", "Tỷ lệ tin", "Số tin", "χ²"]
    assert table.iloc[0]["Kỹ năng"] == stats[0].skill
    # Cùng cách viết số với dữ kiện mà LLM thấy: 1,0 → "100%", 0,5 → "50%".
    assert table.iloc[0]["Tỷ lệ tin"] == fmt_number(stats[0].rate, pct=True)


def test_match_table_explains_a_match_by_shared_skills():
    postings = pd.DataFrame(
        {
            "job_title": ["Kế toán", "Sales"],
            "provinces": [["hà nội"], ["bắc ninh"]],
            "salary_mid": [12.0, None],
        },
        index=pd.Index([7, 9], name="posting_id"),
    )
    skills_of = {7: ["excel", "misa"], 9: ["đàm phán"]}

    table = match_table(postings, skills_of, [9, 7], ["misa", "excel", "python"])

    assert table["Mã tin"].tolist() == [9, 7]
    assert table.iloc[1]["Kỹ năng trùng"] == "excel, misa"
    assert table.iloc[0]["Kỹ năng trùng"] == ""
    assert table.iloc[0]["Lương (triệu)"] == "thoả thuận"


def _answer(small_graph):
    linked = LinkedEntities(skills=["excel"], titles=["kế toán tổng hợp"], provinces=["bắc ninh"])
    facts = [Fact("salary", "x", [12.5], [0, 2])]
    return Answer("...", False, [2], facts, linked, llm_called=True, allowed_ids=[0, 2])


def test_answer_subgraph_keeps_linked_nodes_and_cited_postings(small_graph):
    sub = answer_subgraph(small_graph, _answer(small_graph))

    assert {q.title_id("kế toán tổng hợp"), q.province_id("bắc ninh"), q.skill_id("excel")} <= set(sub)
    assert q.posting_id(2) in sub and q.posting_id(0) not in sub
    assert sub.has_edge(q.posting_id(2), q.title_id("kế toán tổng hợp"))
    assert all(small_graph.has_edge(u, v) for u, v in sub.edges)


def test_subgraph_html_is_a_standalone_page(small_graph):
    html = subgraph_html(answer_subgraph(small_graph, _answer(small_graph)))

    # pyvis ghi nhãn dạng JSON, chữ có dấu thành \uXXXX; trình duyệt vẫn hiện đúng.
    assert "<html>" in html and json.dumps("kế toán tổng hợp")[1:-1] in html


def test_subgraph_html_escapes_markup_in_posting_titles():
    import networkx as nx

    sub = nx.DiGraph()
    sub.add_node(
        "posting:1", type="JobPosting", posting_id=1, title="Kế toán</script><script>alert(1)</script>"
    )

    html = subgraph_html(sub)

    # Tên tin là dữ liệu ngoài và trang nằm trong iframe cùng origin; pyvis mã hoá "<" thành \u003c.
    assert "alert(1)</script>" not in html and "<script>alert" not in html


def test_skills_table_falls_back_to_chi2_with_the_titles_main_category(small_graph):
    stats = q.missing_skills(small_graph, [], q.title_id("kế toán tổng hợp"), k=3)
    assert all(s.chi2 is None for s in stats)  # χ² chỉ tính giữa kỹ năng và nhóm ngành
    category = q.category_id("tài_chính_kế_toán")
    expected = small_graph.edges[q.skill_id(stats[0].skill), category]["chi2"]

    table = skills_table(stats, small_graph, category)

    assert table.iloc[0]["χ²"] == fmt_number(expected)
