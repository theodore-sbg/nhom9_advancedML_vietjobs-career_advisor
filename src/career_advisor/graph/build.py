"""Dựng đồ thị tri thức từ bảng tin sạch và kỹ năng đã gộp tên.

Node: JobPosting, JobTitle, Category, Skill, Qualification, Province, ExperienceLevel, SalaryBand.
Cạnh: REQUIRES, IN_CATEGORY, LOCATED_IN, HAS_TITLE, PAYS, REQUIRES_EXPERIENCE, REQUIRES_EDUCATION,
ALIAS_OF, CO_OCCURS_WITH, ASSOCIATED_WITH, TYPICALLY_REQUIRES.

Mọi số trên cạnh (tỷ lệ, χ², PMI) tính bằng `career_advisor.stats`. Truyền `merges` rỗng thì được
đồ thị không gộp tên, dùng cho ablation.
"""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from career_advisor.cleaning.locations import FOREIGN, UNKNOWN
from career_advisor.graph.query import (
    category_id,
    education_id,
    experience_id,
    posting_id,
    province_id,
    salary_id,
    skill_id,
    title_id,
)
from career_advisor.stats import (
    ALPHA,
    MIN_PAIR_POSTINGS,
    MIN_SKILL_POSTINGS,
    chi2_association,
    cooccurrence,
    group_skill_rates,
    multi_hot,
    resolve_skills,
)

MIN_GRAPH_SKILL_POSTINGS = 5
MIN_RATE_WITH = 3
_CHI2_GROUPS = {"category": category_id, "experience_level": experience_id, "salary_band": salary_id}


def _num(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _add_postings(G: nx.DiGraph, postings: pd.DataFrame) -> None:
    for pid, row in postings.iterrows():
        node = posting_id(pid)
        G.add_node(
            node,
            type="JobPosting",
            posting_id=int(pid),
            category=row["category"],
            title=row.get("job_title_display"),
            salary_min=_num(row["salary_min"]),
            salary_max=_num(row["salary_max"]),
            salary_mid=_num(row["salary_mid"]),
        )
        G.add_node(category_id(row["category"]), type="Category", name=row["category"])
        G.add_edge(node, category_id(row["category"]), rel="IN_CATEGORY")
        for province in row["provinces"]:
            if province not in (UNKNOWN, FOREIGN):
                G.add_node(province_id(province), type="Province", name=province)
                G.add_edge(node, province_id(province), rel="LOCATED_IN")
        if pd.notna(row["job_title_node"]):
            G.add_node(title_id(row["job_title_node"]), type="JobTitle", name=row["job_title_node"])
            G.add_edge(node, title_id(row["job_title_node"]), rel="HAS_TITLE")
        if pd.notna(row["salary_band"]):
            G.add_node(salary_id(row["salary_band"]), type="SalaryBand", name=row["salary_band"])
            G.add_edge(
                node,
                salary_id(row["salary_band"]),
                rel="PAYS",
                min=_num(row["salary_min"]),
                max=_num(row["salary_max"]),
            )
        if pd.notna(row["experience_level"]):
            G.add_node(
                experience_id(row["experience_level"]), type="ExperienceLevel", name=row["experience_level"]
            )
            G.add_edge(node, experience_id(row["experience_level"]), rel="REQUIRES_EXPERIENCE")
        if row["education_level"] != "unspecified":
            G.add_node(
                education_id(row["education_level"]), type="Qualification", name=row["education_level"]
            )
            G.add_edge(node, education_id(row["education_level"]), rel="REQUIRES_EDUCATION")


def build_graph(
    postings: pd.DataFrame,
    mentions: pd.DataFrame,
    merges: pd.DataFrame,
    skill_map: pd.DataFrame,
    min_skill_postings: int = MIN_GRAPH_SKILL_POSTINGS,
    min_chi2_postings: int = MIN_SKILL_POSTINGS,
    min_pair: int = MIN_PAIR_POSTINGS,
    min_rate_with: int = MIN_RATE_WITH,
    alpha: float = ALPHA,
) -> nx.DiGraph:
    G = nx.DiGraph()
    _add_postings(G, postings)

    resolved = resolve_skills(mentions[["posting_id", "skill"]], merges)
    X, skills, ids = multi_hot(resolved, postings.index, min_skill_postings)
    kinds = skill_map.drop_duplicates("canonical").set_index("canonical")["kind"]
    counts = resolved.groupby("skill")["posting_id"].nunique()
    for s in skills:
        G.add_node(
            skill_id(s), type="Skill", name=s, kind=kinds.get(s, "technical"), n_postings=int(counts[s])
        )
    graph_skills = set(skills)
    for pid, s in resolved[resolved["skill"].isin(graph_skills)][["posting_id", "skill"]].itertuples(
        index=False
    ):
        G.add_edge(posting_id(pid), skill_id(s), rel="REQUIRES")

    for alias, canonical, tier, score in merges[["skill", "canonical", "tier", "score"]].itertuples(
        index=False
    ):
        if canonical in graph_skills:
            G.add_node(skill_id(alias), type="Skill", name=alias, alias=True)
            G.add_edge(skill_id(alias), skill_id(canonical), rel="ALIAS_OF", tier=tier, score=float(score))
    # Chuỗi gốc → kỹ năng sau tầng luật, để nối tên người dùng gõ ("ReactJS") vào node.
    G.graph["raw_to_skill"] = {
        str(r).strip().lower(): c for r, c in zip(skill_map["raw"], skill_map["canonical"], strict=True)
    }

    for pair in cooccurrence(X, skills, min_pair).itertuples(index=False):
        G.add_edge(
            skill_id(pair.skill_a),
            skill_id(pair.skill_b),
            rel="CO_OCCURS_WITH",
            count=int(pair.count),
            pmi=float(pair.pmi),
            npmi=float(pair.npmi),
        )

    X_chi2, skills_chi2, _ = multi_hot(resolved, postings.index, min_chi2_postings)
    for column, node_of in _CHI2_GROUPS.items():
        result = chi2_association(X_chi2, skills_chi2, postings.loc[ids, column], alpha=alpha)
        for r in result[result["positive"] & result["significant"]].itertuples(index=False):
            G.add_edge(
                skill_id(r.skill),
                node_of(r.group),
                rel="ASSOCIATED_WITH",
                chi2=float(r.chi2),
                p_value=float(r.p_value),
                rate_in_group=float(r.rate_in_group),
                rate_outside=float(r.rate_outside),
                n_with=int(r.n_with_skill_in_group),
            )

    for column, node_of in (("category", category_id), ("job_title_node", title_id)):
        rates = group_skill_rates(X, skills, postings.loc[ids, column], min_with=min_rate_with)
        for r in rates.itertuples(index=False):
            G.add_edge(
                node_of(r.group),
                skill_id(r.skill),
                rel="TYPICALLY_REQUIRES",
                rate=float(r.rate),
                n_with=int(r.n_with),
                n_group=int(r.n_group),
            )

    G.graph["n_postings"] = len(postings)
    return G
