"""Truy vấn đồ thị. Mọi hàm trả kèm `posting_id` làm dẫn chứng và số tin dùng để tính (n)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from career_advisor.cleaning.skills import canonical_skill

EVIDENCE = 5


def posting_id(pid: int) -> str:
    return f"posting:{int(pid)}"


def skill_id(name: str) -> str:
    return f"skill:{name}"


def title_id(name: str) -> str:
    return f"title:{name}"


def category_id(name: str) -> str:
    return f"category:{name}"


def province_id(name: str) -> str:
    return f"province:{name}"


def experience_id(level: str) -> str:
    return f"exp:{level}"


def salary_id(band: str) -> str:
    return f"salary:{band}"


def education_id(level: str) -> str:
    return f"edu:{level}"


@dataclass
class SalarySummary:
    median: float | None
    q1: float | None
    q3: float | None
    n: int
    posting_ids: list[int]


@dataclass
class SkillStat:
    skill: str
    rate: float
    n_with: int
    n_group: int
    chi2: float | None
    posting_ids: list[int]


@dataclass
class RelatedSkill:
    skill: str
    count: int
    pmi: float
    npmi: float


def postings_of(G: nx.DiGraph, node: str) -> set[int]:
    """Các tin nối tới `node` (nhóm ngành, tỉnh, chức danh, kỹ năng…)."""
    if not G.has_node(node):
        return set()
    return {G.nodes[u]["posting_id"] for u in G.predecessors(node) if G.nodes[u]["type"] == "JobPosting"}


def matching_postings(G: nx.DiGraph, *nodes: str) -> set[int]:
    """Các tin nối tới tất cả `nodes` cùng lúc."""
    sets = [postings_of(G, n) for n in nodes]
    return set.intersection(*sets) if sets else set()


def salary_summary(G: nx.DiGraph, *nodes: str) -> SalarySummary:
    """Lương (triệu đồng, mức giữa) của các tin khớp mọi điều kiện. Bỏ tin lương thoả thuận."""
    salaries = {
        pid: G.nodes[posting_id(pid)]["salary_mid"]
        for pid in matching_postings(G, *nodes)
        if G.nodes[posting_id(pid)]["salary_mid"] is not None
    }
    if not salaries:
        return SalarySummary(None, None, None, 0, [])
    values = np.array(list(salaries.values()))
    median = float(np.median(values))
    closest = sorted(salaries, key=lambda pid: (abs(salaries[pid] - median), pid))[:EVIDENCE]
    return SalarySummary(
        median=median,
        q1=float(np.percentile(values, 25)),
        q3=float(np.percentile(values, 75)),
        n=len(values),
        posting_ids=closest,
    )


def resolve_skill(G: nx.DiGraph, name: str) -> str | None:
    """Tên người dùng gõ → tên kỹ năng trong đồ thị. Đi qua chuỗi gốc, luật, rồi cạnh ALIAS_OF."""
    raw_to_skill = G.graph.get("raw_to_skill", {})
    for candidate in (raw_to_skill.get(name.strip().lower()), canonical_skill(name)):
        if not candidate or not G.has_node(skill_id(candidate)):
            continue
        node = skill_id(candidate)
        for _, target, data in G.out_edges(node, data=True):
            if data["rel"] == "ALIAS_OF":
                node = target
        return G.nodes[node]["name"]
    return None


def top_skills(G: nx.DiGraph, group: str, k: int = 10) -> list[SkillStat]:
    """Kỹ năng hay được yêu cầu nhất của một chức danh hoặc nhóm ngành, xếp theo tỷ lệ tin."""
    if not G.has_node(group):
        return []
    edges = [(v, d) for _, v, d in G.out_edges(group, data=True) if d["rel"] == "TYPICALLY_REQUIRES"]
    edges.sort(key=lambda e: (-e[1]["rate"], G.nodes[e[0]]["name"]))
    in_group = postings_of(G, group)
    result = []
    for skill_node, d in edges[:k]:
        association = G.get_edge_data(skill_node, group)
        chi2 = association["chi2"] if association and association["rel"] == "ASSOCIATED_WITH" else None
        evidence = sorted(in_group & postings_of(G, skill_node))[:EVIDENCE]
        result.append(
            SkillStat(G.nodes[skill_node]["name"], d["rate"], d["n_with"], d["n_group"], chi2, evidence)
        )
    return result


def missing_skills(G: nx.DiGraph, have: list[str], group: str, k: int = 10) -> list[SkillStat]:
    """Kỹ năng hay được yêu cầu của `group` mà người dùng chưa có. Tên người dùng gõ được nối qua alias."""
    owned = {resolved for h in have if (resolved := resolve_skill(G, h))}
    candidates = top_skills(G, group, k=k + len(owned))
    return [s for s in candidates if s.skill not in owned][:k]


def related_skills(G: nx.DiGraph, name: str, k: int = 10) -> list[RelatedSkill]:
    """Kỹ năng hay đi cùng, xếp theo số tin cùng xuất hiện rồi tới NPMI."""
    skill = resolve_skill(G, name)
    if skill is None:
        return []
    node = skill_id(skill)
    edges = [(v, d) for _, v, d in G.out_edges(node, data=True) if d["rel"] == "CO_OCCURS_WITH"]
    edges += [(u, d) for u, _, d in G.in_edges(node, data=True) if d["rel"] == "CO_OCCURS_WITH"]
    edges.sort(key=lambda e: (-e[1]["count"], -e[1]["npmi"], G.nodes[e[0]]["name"]))
    return [RelatedSkill(G.nodes[v]["name"], d["count"], d["pmi"], d["npmi"]) for v, d in edges[:k]]
