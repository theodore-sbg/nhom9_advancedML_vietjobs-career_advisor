"""Bộ câu hỏi kiểm thử cho KG-RAG. Đáp án tính bằng pandas trên bảng sạch, độc lập với đồ thị.

3 nhóm:
- one_hop: một điều kiện hoặc một phép tra (lương theo chức danh, tỷ lệ kỹ năng…);
- multi_hop: nhiều điều kiện hoặc nhiều bước (kỹ năng còn thiếu + lương, lương theo 3 điều kiện…);
- out_of_scope: dữ liệu không trả lời được, hệ thống phải từ chối.
Tổ hợp có ít hơn `min_n` tin không dùng làm câu có đáp án; chúng thành câu từ chối.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_advisor.cleaning.locations import FOREIGN, UNKNOWN
from career_advisor.config import SEED

MIN_N = 5
TOP_MISSING = 3

COUNTS = {
    "salary_title": 6,
    "salary_category_province": 5,
    "skill_rate": 5,
    "count_title_province": 5,
    "cooccur": 4,
    "gap_and_salary": 6,
    "salary_3way": 5,
    "best_paid_title": 4,
    "unknown_title": 3,
    "foreign": 2,
    "out_of_time": 2,
    "too_few_postings": 3,
}
GROUP_OF = {
    "salary_title": "one_hop",
    "salary_category_province": "one_hop",
    "skill_rate": "one_hop",
    "count_title_province": "one_hop",
    "cooccur": "one_hop",
    "gap_and_salary": "multi_hop",
    "salary_3way": "multi_hop",
    "best_paid_title": "multi_hop",
    "unknown_title": "out_of_scope",
    "foreign": "out_of_scope",
    "out_of_time": "out_of_scope",
    "too_few_postings": "out_of_scope",
}
# Nghề không có trong dữ liệu (đã kiểm không trùng chức danh nào khi sinh câu hỏi).
UNKNOWN_TITLES = ("phi công", "nhà du hành vũ trụ", "thợ lặn biển sâu", "bác sĩ phẫu thuật tim")
FOREIGN_PLACES = ("Nhật Bản", "Singapore", "Đức")


def small_counts() -> dict[str, int]:
    """Mỗi loại 1 câu, dùng cho test với dữ liệu nhỏ."""
    return dict.fromkeys(COUNTS, 1)


def _has_province(postings: pd.DataFrame, province: str) -> pd.Series:
    # Chỉ mục tỉnh → tin dựng 1 lần rồi giữ trong attrs, vì hàm này được gọi cho hàng nghìn tổ hợp.
    index = postings.attrs.get("province_index")
    if index is None:
        exploded = postings["provinces"].explode()
        index = exploded.groupby(exploded).groups
        postings.attrs["province_index"] = index
    return pd.Series(postings.index.isin(index.get(province, [])), index=postings.index)


def _subset(postings: pd.DataFrame, **filters: str) -> pd.DataFrame:
    mask = pd.Series(True, index=postings.index)
    for column, value in filters.items():
        mask &= _has_province(postings, value) if column == "province" else postings[column] == value
    return postings[mask]


def salary_answer(postings: pd.DataFrame, **filters: str) -> dict:
    paid = _subset(postings, **filters)["salary_mid"].dropna()
    return {
        "median": float(paid.median()) if len(paid) else None,
        "n": len(paid),
        "posting_ids": sorted(paid.index),
    }


def skill_rate_answer(postings: pd.DataFrame, skills: pd.DataFrame, title: str, skill: str) -> dict:
    group = set(_subset(postings, job_title_node=title).index)
    with_skill = set(skills.loc[skills["skill"] == skill, "posting_id"]) & group
    return {
        "rate": len(with_skill) / len(group),
        "n_with": len(with_skill),
        "n_group": len(group),
        "n": len(group),
        "posting_ids": sorted(with_skill),
    }


def cooccur_answer(skills: pd.DataFrame, skill: str) -> dict:
    ids = set(skills.loc[skills["skill"] == skill, "posting_id"])
    others = skills[skills["posting_id"].isin(ids) & (skills["skill"] != skill)]
    counts = others["skill"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    if counts.empty:
        return {"skill": None, "count": 0, "unique_top": False, "n": len(ids), "posting_ids": []}
    top = counts.index[0]
    return {
        "skill": top,
        "count": int(counts.iloc[0]),
        "unique_top": bool(len(counts) == 1 or counts.iloc[0] > counts.iloc[1]),
        "n": len(ids),
        "posting_ids": sorted(set(others.loc[others["skill"] == top, "posting_id"])),
    }


def _title_rates(postings: pd.DataFrame, skills: pd.DataFrame, title: str) -> pd.Series:
    group = set(_subset(postings, job_title_node=title).index)
    in_group = skills[skills["posting_id"].isin(group)]
    rates = in_group.groupby("skill")["posting_id"].nunique() / len(group)
    return rates.sort_index().sort_values(ascending=False, kind="stable")


def gap_answer(postings: pd.DataFrame, skills: pd.DataFrame, title: str, have: list[str]) -> dict:
    rates = _title_rates(postings, skills, title)
    missing = [s for s in rates.index if s not in have][:TOP_MISSING]
    group = _subset(postings, job_title_node=title)
    return {"missing": missing, "rates": [float(rates[s]) for s in missing], "n_group": len(group)}


def best_paid_title_answer(postings: pd.DataFrame, category: str, min_postings: int = MIN_N) -> dict:
    paid = _subset(postings, category=category).dropna(subset=["salary_mid", "job_title_node"])
    stats = paid.groupby("job_title_node")["salary_mid"].agg(["median", "size"])
    stats = (
        stats[stats["size"] >= min_postings]
        .sort_index()
        .sort_values("median", ascending=False, kind="stable")
    )
    title = stats.index[0]
    return {
        "title": title,
        "median": float(stats.loc[title, "median"]),
        "n": int(stats.loc[title, "size"]),
        "n_titles": len(stats),
        "posting_ids": sorted(paid.index[paid["job_title_node"] == title]),
    }


def _pretty(text: str) -> str:
    return text.replace("_", " ")


def _place(province: str) -> str:
    return province.title()


def _real_provinces(postings: pd.DataFrame) -> list[str]:
    return sorted({p for ps in postings["provinces"] for p in ps} - {UNKNOWN, FOREIGN})


def build_question_set(
    postings: pd.DataFrame,
    skills: pd.DataFrame,
    counts: dict[str, int] = COUNTS,
    min_n: int = MIN_N,
    seed: int = SEED,
) -> list[dict]:
    """Sinh bộ câu hỏi. `skills` có cột posting_id, skill (đã gộp tên)."""
    rng = np.random.default_rng(seed)
    display = (
        postings.dropna(subset=["job_title_node"]).groupby("job_title_node")["job_title_display"].first()
    )
    titles = sorted(display.index)
    provinces = _real_provinces(postings)
    categories = sorted(postings["category"].unique())
    out: list[dict] = []

    def pick(candidates: list, k: int) -> list:
        if not candidates:
            return []
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        return [candidates[i] for i in sorted(idx)]

    def add(kind: str, question: str, params: dict, answer: dict) -> None:
        ids = answer.pop("posting_ids", [])
        out.append(
            {
                "type": kind,
                "question": question,
                "params": params,
                "answer": answer,
                "valid_posting_ids": [int(i) for i in ids],
            }
        )

    # --- one_hop ---
    ok = [t for t in titles if salary_answer(postings, job_title_node=t)["n"] >= min_n]
    for t in pick(ok, counts["salary_title"]):
        add(
            "salary_title",
            f"Lương trung vị của {display[t]} là bao nhiêu?",
            {"title": t},
            salary_answer(postings, job_title_node=t),
        )

    combos = [(c, p) for c in categories for p in provinces]
    ok = [(c, p) for c, p in combos if salary_answer(postings, category=c, province=p)["n"] >= min_n]
    for c, p in pick(ok, counts["salary_category_province"]):
        add(
            "salary_category_province",
            f"Lương trung vị của nhóm ngành {_pretty(c)} ở {_place(p)} là bao nhiêu?",
            {"category": c, "province": p},
            salary_answer(postings, category=c, province=p),
        )

    rate_pairs = []
    for t in titles:
        if len(_subset(postings, job_title_node=t)) >= min_n:
            rate_pairs += [(t, s) for s in _title_rates(postings, skills, t).index[:10]]
    for t, s in pick(rate_pairs, counts["skill_rate"]):
        add(
            "skill_rate",
            f"Bao nhiêu phần trăm tin tuyển {display[t]} yêu cầu kỹ năng {s}?",
            {"title": t, "skill": s},
            skill_rate_answer(postings, skills, t, s),
        )

    count_pairs = [
        (t, p)
        for t in titles
        for p in provinces
        if len(_subset(postings, job_title_node=t, province=p)) >= min_n
    ]
    for t, p in pick(count_pairs, counts["count_title_province"]):
        subset = _subset(postings, job_title_node=t, province=p)
        add(
            "count_title_province",
            f"Có bao nhiêu tin tuyển {display[t]} ở {_place(p)}?",
            {"title": t, "province": p},
            {"count": len(subset), "n": len(subset), "posting_ids": sorted(subset.index)},
        )

    frequent = skills["skill"].value_counts()
    frequent = sorted(frequent[frequent >= min_n].index)
    unique_top = [s for s in frequent if cooccur_answer(skills, s)["unique_top"]]
    for s in pick(unique_top, counts["cooccur"]):
        add(
            "cooccur",
            f"Kỹ năng nào hay đi cùng {s} nhất trong các tin tuyển dụng?",
            {"skill": s},
            cooccur_answer(skills, s),
        )

    # --- multi_hop ---
    gap = [
        (t, p) for t, p in count_pairs if salary_answer(postings, job_title_node=t, province=p)["n"] >= min_n
    ]
    for t, p in pick(gap, counts["gap_and_salary"]):
        top = list(_title_rates(postings, skills, t).index[:5])
        have = [top[i] for i in sorted(rng.choice(len(top), size=min(2, len(top)), replace=False))]
        answer = gap_answer(postings, skills, t, have)
        salary = salary_answer(postings, job_title_node=t, province=p)
        answer.update(
            {"salary_median": salary["median"], "n": salary["n"], "posting_ids": salary["posting_ids"]}
        )
        add(
            "gap_and_salary",
            f"Tôi biết {' và '.join(have)}, muốn làm {display[t]} ở {_place(p)}. "
            f"Tôi còn thiếu những kỹ năng nào quan trọng nhất, và lương trung vị khoảng bao nhiêu?",
            {"title": t, "province": p, "have": have},
            answer,
        )

    levels = sorted(postings["experience_level"].dropna().unique())
    three = [
        (t, p, e)
        for t, p in gap
        for e in levels
        if salary_answer(postings, job_title_node=t, province=p, experience_level=e)["n"] >= min_n
    ]
    for t, p, e in pick(three, counts["salary_3way"]):
        add(
            "salary_3way",
            f"Làm {display[t]} ở {_place(p)} với mức kinh nghiệm {e.replace('_', ' ')}, "
            f"lương trung vị là bao nhiêu?",
            {"title": t, "province": p, "experience_level": e},
            salary_answer(postings, job_title_node=t, province=p, experience_level=e),
        )

    ok = []
    for c in categories:
        try:
            if best_paid_title_answer(postings, c, min_n)["n_titles"] >= 2:
                ok.append(c)
        except IndexError:
            continue
    for c in pick(ok, counts["best_paid_title"]):
        add(
            "best_paid_title",
            f"Trong nhóm ngành {_pretty(c)}, chức danh nào có lương trung vị cao nhất "
            f"(chỉ tính chức danh có từ {min_n} tin ghi lương)?",
            {"category": c},
            best_paid_title_answer(postings, c, min_n),
        )

    # --- out_of_scope ---
    norm_titles = " | ".join(postings["job_title_node"].dropna().unique())
    unknown = [t for t in UNKNOWN_TITLES if t not in norm_titles]
    for t in pick(unknown, counts["unknown_title"]):
        add("unknown_title", f"Lương trung vị của {t} ở Hà Nội là bao nhiêu?", {"title": t}, {"refuse": True})
    for place in pick(list(FOREIGN_PLACES), counts["foreign"]):
        add("foreign", f"Lương kế toán ở {place} là bao nhiêu?", {"place": place}, {"refuse": True})
    for t in pick(titles, counts["out_of_time"]):
        add(
            "out_of_time",
            f"Lương trung vị của {display[t]} năm 2023 là bao nhiêu?",
            {"title": t},
            {"refuse": True},
        )
    small = [
        (t, p)
        for t in titles
        for p in provinces
        if 1 <= salary_answer(postings, job_title_node=t, province=p)["n"] < min_n
    ]
    for t, p in pick(small, counts["too_few_postings"]):
        add(
            "too_few_postings",
            f"Lương trung vị của {display[t]} ở {_place(p)} là bao nhiêu?",
            {"title": t, "province": p},
            {"refuse": True},
        )

    for q in out:
        q["group"] = GROUP_OF[q["type"]]
    for group in ("one_hop", "multi_hop", "out_of_scope"):
        members = [q for q in out if q["group"] == group]
        for k, i in enumerate(rng.permutation(len(members))):
            members[i]["split"] = "dev" if k % 2 == 0 else "test"
    for k, q in enumerate(out):
        q["id"] = f"q{k + 1:02d}"
    return [
        {
            key: q[key]
            for key in ("id", "group", "type", "split", "question", "params", "answer", "valid_posting_ids")
        }
        for q in out
    ]
