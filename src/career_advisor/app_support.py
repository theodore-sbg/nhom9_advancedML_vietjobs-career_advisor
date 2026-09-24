"""Các hàm thuần cho giao diện Streamlit: bảng kỹ năng, bảng tin phù hợp, đồ thị con quanh câu trả lời."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from career_advisor.graph import query as q
from career_advisor.rag.answer import Answer
from career_advisor.rag.subgraph import fmt_number, group_node

MAX_POSTINGS = 8
COLORS = {
    "JobPosting": "#9aa5b1",
    "JobTitle": "#e8833a",
    "Category": "#c0504d",
    "Skill": "#4f81bd",
    "Province": "#9bbb59",
    "ExperienceLevel": "#8064a2",
}


def skills_table(
    stats: list[q.SkillStat], G: nx.DiGraph | None = None, category: str | None = None
) -> pd.DataFrame:
    """χ² chỉ tính giữa kỹ năng và nhóm ngành. Với chức danh, lấy χ² theo nhóm ngành chính của nó.
    Không có cạnh nghĩa là liên hệ không dương hoặc không có ý nghĩa thống kê: hiện "–"."""

    def chi2(s: q.SkillStat) -> float | None:
        if s.chi2 is not None or G is None or category is None:
            return s.chi2
        edge = G.get_edge_data(q.skill_id(s.skill), category)
        return edge["chi2"] if edge and edge["rel"] == "ASSOCIATED_WITH" else None

    values = [chi2(s) for s in stats]
    return pd.DataFrame(
        {
            "Kỹ năng": [s.skill for s in stats],
            "Tỷ lệ tin": [fmt_number(s.rate, pct=True) for s in stats],
            "Số tin": [f"{s.n_with}/{s.n_group}" for s in stats],
            "χ²": [fmt_number(v) if v is not None else "–" for v in values],
        }
    )


def match_table(
    postings: pd.DataFrame, skills_of: dict[int, list[str]], ids: list[int], cv_skills: list[str]
) -> pd.DataFrame:
    """Tin phù hợp theo thứ tự truy xuất. Lý do: kỹ năng của CV có trong tin."""
    have = set(cv_skills)
    rows = []
    for pid in ids:
        row = postings.loc[pid]
        salary = row["salary_mid"]
        rows.append(
            {
                "Mã tin": pid,
                "Chức danh": row["job_title"],
                "Tỉnh": ", ".join(p.title() for p in row["provinces"]),
                "Lương (triệu)": "thoả thuận" if pd.isna(salary) else fmt_number(salary),
                "Kỹ năng trùng": ", ".join(sorted(have & set(skills_of.get(pid, [])))),
            }
        )
    return pd.DataFrame(rows)


def answer_subgraph(G: nx.DiGraph, answer: Answer, max_postings: int = MAX_POSTINGS) -> nx.DiGraph:
    """Node đã nối (nghề, tỉnh, kinh nghiệm, kỹ năng) và các tin được trích, cùng cạnh giữa chúng."""
    linked = answer.linked
    nodes = []
    if linked:
        group = group_node(linked)
        nodes += [group] if group else []
        nodes += [q.province_id(p) for p in linked.provinces]
        nodes += [q.experience_id(e) for e in linked.experience_levels]
        nodes += [q.skill_id(s) for s in linked.skills]
    nodes += [q.posting_id(pid) for pid in answer.citations[:max_postings]]
    return G.subgraph(n for n in nodes if G.has_node(n)).copy()


def _label(G: nx.DiGraph, node: str) -> str:
    data = G.nodes[node]
    if data["type"] == "JobPosting":
        return f"#{data['posting_id']} {data['title']}"
    return data["name"].replace("_", " ")


def subgraph_html(sub: nx.DiGraph, height: str = "480px") -> str:
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line")
    for node, data in sub.nodes(data=True):
        net.add_node(
            node, label=_label(sub, node), color=COLORS.get(data["type"], "#cccccc"), title=data["type"]
        )
    for u, v, data in sub.edges(data=True):
        net.add_edge(u, v, title=data["rel"])
    html = net.generate_html()
    # Thu cả đồ thị vào khung khi mô phỏng lực đã dừng.
    return html.replace(
        "return network;",
        'network.once("stabilizationIterationsDone", () => network.fit());\n    return network;',
        1,
    )
