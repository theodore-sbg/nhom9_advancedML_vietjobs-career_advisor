"""Lấy dữ kiện từ đồ thị con quanh các node đã nối. Mọi con số được tính bằng code ở đây.

Mỗi `Fact` có câu mô tả (con số đã định dạng kiểu Việt), danh sách con số (`values`) để Verifier đối
chiếu, và `posting_id` làm dẫn chứng. LLM chỉ được diễn đạt lại các dữ kiện này.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from career_advisor.graph import query as q
from career_advisor.rag.linking import DATA_YEAR, LinkedEntities

MIN_N = 5
TOP_SKILLS = 8
TOP_MISSING = 5
TOP_RELATED = 5
MAX_LINKED_SKILLS = 3
TOP_TITLES = 5
EVIDENCE = 5
EXPERIENCE_TEXT = {
    "none": "không yêu cầu kinh nghiệm",
    "under_1y": "dưới 1 năm kinh nghiệm",
    "1_2y": "1–2 năm kinh nghiệm",
    "3_4y": "3–4 năm kinh nghiệm",
    "5y_plus": "từ 5 năm kinh nghiệm",
}


@dataclass
class Fact:
    kind: str
    text: str
    values: list[float] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)


def fmt_number(x: float, pct: bool = False) -> str:
    """12.5 → "12,5"; 12.0 → "12"; 0.509 (pct) → "50,9%"."""
    value = round(x * 100 if pct else x, 1)
    text = f"{value:.1f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{text}%" if pct else text


def _scope_text(G: nx.DiGraph, nodes: list[str]) -> str:
    parts = []
    for node in nodes:
        data = G.nodes[node]
        name = data["name"]
        parts.append(
            {
                "JobTitle": f"{name}",
                "Category": f"nhóm ngành {name.replace('_', ' ')}",
                "Province": f"ở {name.title()}",
                "ExperienceLevel": f"với {EXPERIENCE_TEXT.get(name, name)}",
            }.get(data["type"], name)
        )
    return " ".join(parts)


def _salary_facts(G: nx.DiGraph, nodes: list[str]) -> list[Fact]:
    scope = _scope_text(G, nodes)
    everything = sorted(q.matching_postings(G, *nodes))
    summary = q.salary_summary(G, *nodes)
    if summary.n < MIN_N:
        return [
            Fact(
                "insufficient",
                f"Dữ liệu chỉ có {summary.n} tin ghi lương cho {scope} (cần ít nhất {MIN_N} tin), "
                "nên không đủ để nêu mức lương.",
                [summary.n],
            )
        ]
    return [
        Fact(
            "salary",
            f"Lương trung vị của {scope}: {fmt_number(summary.median)} triệu đồng/tháng "
            f"(khoảng giữa {fmt_number(summary.q1)}–{fmt_number(summary.q3)} triệu), "
            f"tính trên {summary.n} tin có ghi lương.",
            [summary.median, summary.q1, summary.q3, summary.n],
            summary.posting_ids,
        ),
        Fact("count", f"Có {len(everything)} tin tuyển {scope}.", [len(everything)], everything[:EVIDENCE]),
    ]


def _skill_list_fact(G: nx.DiGraph, kind: str, intro: str, stats: list[q.SkillStat]) -> Fact | None:
    if not stats:
        return None

    def label(s: q.SkillStat) -> str:
        soft = G.nodes[q.skill_id(s.skill)].get("kind") == "soft"
        return f"{s.skill} ({fmt_number(s.rate, pct=True)} tin{', kỹ năng mềm' if soft else ''})"

    listing = ", ".join(label(s) for s in stats)
    sources = sorted({pid for s in stats for pid in s.posting_ids[:2]})
    return Fact(
        kind,
        f"{intro} (trên {stats[0].n_group} tin): {listing}.",
        [round(s.rate * 100, 1) for s in stats] + [stats[0].n_group],
        sources,
    )


def _best_paid_titles(G: nx.DiGraph, category: str) -> Fact | None:
    rows = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "JobTitle":
            summary = q.salary_summary(G, node, category)
            if summary.n >= MIN_N:
                rows.append((summary.median, data["name"], summary))
    if not rows:
        return None
    rows.sort(key=lambda r: (-r[0], r[1]))
    top = rows[:TOP_TITLES]
    listing = ", ".join(f"{name} ({fmt_number(m)} triệu, {s.n} tin)" for m, name, s in top)
    return Fact(
        "best_paid_titles",
        f"Chức danh lương trung vị cao nhất trong {_scope_text(G, [category])} "
        f"(từ {MIN_N} tin ghi lương): {listing}.",
        [v for m, _, s in top for v in (m, s.n)],
        [pid for _, _, s in top for pid in s.posting_ids[:1]],
    )


def retrieve_facts(G: nx.DiGraph, linked: LinkedEntities) -> list[Fact]:
    if linked.foreign or linked.other_years:
        reason = (
            "nơi làm ở nước ngoài" if linked.foreign else f"năm {', '.join(map(str, linked.other_years))}"
        )
        return [
            Fact(
                "out_of_scope",
                f"Câu hỏi nói về {reason}, nhưng dữ liệu chỉ có tin đăng trên TopCV tại Việt Nam "
                f"từ tháng 7 đến tháng 10/{DATA_YEAR}.",
            )
        ]

    group = (
        q.title_id(linked.titles[0])
        if linked.titles
        else q.category_id(linked.categories[0])
        if linked.categories
        else None
    )
    filters = [q.province_id(p) for p in linked.provinces[:1]]
    filters += [q.experience_id(e) for e in linked.experience_levels[:1]]
    facts: list[Fact] = []
    if group or filters:
        facts += _salary_facts(G, [n for n in [group, *filters] if n])

    if group:
        top = q.top_skills(G, group, k=TOP_SKILLS)
        fact = _skill_list_fact(
            G, "top_skills", f"Kỹ năng hay được yêu cầu nhất của {_scope_text(G, [group])}", top
        )
        facts += [fact] if fact else []
        if linked.skills:
            missing = q.missing_skills(G, linked.skills, group, k=TOP_MISSING)
            intro = (
                f"Kỹ năng người hỏi còn thiếu so với {_scope_text(G, [group])}, xếp theo tỷ lệ tin yêu cầu"
            )
            fact = _skill_list_fact(G, "missing_skills", intro, missing)
            facts += [fact] if fact else []
    if linked.categories and not linked.titles:
        fact = _best_paid_titles(G, q.category_id(linked.categories[0]))
        facts += [fact] if fact else []

    for skill in linked.skills[:MAX_LINKED_SKILLS]:
        related = q.related_skills(G, skill, k=TOP_RELATED)
        if related:
            listing = ", ".join(f"{r.skill} ({r.count} tin)" for r in related)
            both = q.matching_postings(G, q.skill_id(skill), q.skill_id(related[0].skill))
            sources = sorted(both)[:EVIDENCE]
            facts.append(
                Fact(
                    "related_skills",
                    f"Kỹ năng hay đi cùng {skill}: {listing}.",
                    [r.count for r in related],
                    sources,
                )
            )
    return facts
