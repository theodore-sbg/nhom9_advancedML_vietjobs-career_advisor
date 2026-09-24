"""Chấm lộ trình học (ablation 1 agent so với nhiều agent) trên bộ CV mẫu.

Không có đáp án lộ trình chuẩn, nên đo các điều kiểm được bằng code:
- nghề mà hệ nối được có thuộc đúng nhóm ngành của CV không;
- lộ trình có nhắc tới các kỹ năng còn thiếu hàng đầu theo đồ thị không.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

import networkx as nx

from career_advisor.cleaning.locations import tone_new_style
from career_advisor.graph import query as q


def _norm(text: str) -> str:
    return tone_new_style(unicodedata.normalize("NFC", text).lower())


def mentioned_share(text: str, skills: list[str]) -> float | None:
    if not skills:
        return None
    norm = _norm(text)
    return sum(_norm(s) in norm for s in skills) / len(skills)


def dominant_category(G: nx.DiGraph, node: str) -> str | None:
    """Nhóm ngành chiếm nhiều tin nhất của một chức danh; nhóm ngành thì là chính nó."""
    if G.nodes[node]["type"] == "Category":
        return G.nodes[node]["name"]
    counts = Counter(
        G.nodes[c]["name"]
        for pid in q.postings_of(G, node)
        for _, c, d in G.out_edges(q.posting_id(pid), data=True)
        if d["rel"] == "IN_CATEGORY"
    )
    return counts.most_common(1)[0][0] if counts else None
