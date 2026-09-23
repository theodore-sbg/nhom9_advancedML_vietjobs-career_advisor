import numpy as np
import pandas as pd
import pytest

from career_advisor.graph import query as q
from career_advisor.graph.build import build_graph

SMALL = {"min_skill_postings": 1, "min_chi2_postings": 1, "min_pair": 1, "min_rate_with": 1, "alpha": 1.0}


def _postings() -> pd.DataFrame:
    rows = [
        # category, provinces, salary_mid, band, exp, edu, title
        ("it", ["hà nội"], 20.0, "20_30m", "1_2y", "university", "lập trình viên"),
        ("it", ["hà nội"], 30.0, "30_50m", "3_4y", "university", "lập trình viên"),
        ("it", ["hồ chí minh"], 25.0, "20_30m", "1_2y", "college", "lập trình viên"),
        ("it", ["hà nội", "hồ chí minh"], None, None, "none", "unspecified", None),
        ("acc", ["hà nội"], 10.0, "10_15m", "1_2y", "college", "kế toán"),
        ("acc", ["hà nội"], 12.0, "10_15m", "none", "college", "kế toán"),
        ("acc", ["bắc ninh"], 14.0, "10_15m", "3_4y", "vocational", "kế toán"),
        ("acc", ["unknown"], 16.0, "15_20m", "1_2y", "college", None),
    ]
    df = pd.DataFrame(
        rows,
        columns=["category", "provinces", "salary_mid", "salary_band", "experience_level", "education_level",
                 "job_title_node"],
    )  # fmt: skip
    df["salary_min"] = df["salary_mid"] - 2
    df["salary_max"] = df["salary_mid"] + 2
    df["job_title_display"] = df["job_title_node"].str.title()
    df.index.name = "posting_id"
    return df


SKILLS = {
    0: ["python", "git"],
    1: ["python", "git", "docker"],
    2: ["python", "reactjs"],
    3: ["git"],
    4: ["excel", "misa"],
    5: ["excel", "misa", "giao tiếp"],
    6: ["excel"],
    7: ["excel", "python"],
}


def _mentions() -> pd.DataFrame:
    return pd.DataFrame(
        [(pid, s, "soft" if s == "giao tiếp" else "technical") for pid, ss in SKILLS.items() for s in ss],
        columns=["posting_id", "skill", "source"],
    )


MERGES = pd.DataFrame({"skill": ["reactjs"], "canonical": ["react"], "tier": ["embedding"], "score": [0.97]})
SKILL_MAP = pd.DataFrame(
    {
        "raw": ["Python", "Excel", "MISA", "ReactJS", "Git", "Docker", "Giao tiếp tốt"],
        "canonical": ["python", "excel", "misa", "reactjs", "git", "docker", "giao tiếp"],
        "kind": ["technical"] * 6 + ["soft"],
        "tier": ["rule"] * 7,
    }
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(_postings(), _mentions(), MERGES, SKILL_MAP, **SMALL)


def test_graph_has_every_node_type(graph):
    types = {d["type"] for _, d in graph.nodes(data=True)}

    assert types == {
        "JobPosting", "JobTitle", "Category", "Skill", "Province", "ExperienceLevel", "SalaryBand",
        "Qualification",
    }  # fmt: skip


def test_graph_has_every_edge_type(graph):
    rels = {d["rel"] for _, _, d in graph.edges(data=True)}

    assert rels == {
        "REQUIRES", "IN_CATEGORY", "LOCATED_IN", "HAS_TITLE", "PAYS", "REQUIRES_EXPERIENCE",
        "REQUIRES_EDUCATION", "ALIAS_OF", "CO_OCCURS_WITH", "ASSOCIATED_WITH", "TYPICALLY_REQUIRES",
    }  # fmt: skip


def test_unknown_province_and_negotiable_salary_have_no_edge(graph):
    assert not graph.has_node(q.province_id("unknown"))
    assert not any(d["rel"] == "PAYS" for _, _, d in graph.out_edges(q.posting_id(3), data=True))


def test_merged_skill_becomes_alias_of_canonical(graph):
    assert graph.has_edge(q.skill_id("reactjs"), q.skill_id("react"))
    assert q.resolve_skill(graph, "ReactJS") == "react"
    assert q.resolve_skill(graph, "Giao tiếp tốt") == "giao tiếp"
    assert q.resolve_skill(graph, "không có") is None
    assert graph.has_edge(q.posting_id(2), q.skill_id("react"))


def test_salary_summary_matches_pandas(graph):
    postings = _postings()
    expected = postings[postings.category == "acc"].salary_mid.dropna()

    summary = q.salary_summary(graph, q.category_id("acc"))

    assert summary.n == len(expected) == 4
    assert summary.median == pytest.approx(expected.median())
    assert summary.q1 == pytest.approx(np.percentile(expected, 25))
    assert summary.q3 == pytest.approx(np.percentile(expected, 75))
    assert set(summary.posting_ids) <= set(expected.index)


def test_salary_summary_intersects_filters(graph):
    summary = q.salary_summary(graph, q.category_id("it"), q.province_id("hà nội"))

    # tin 0 và 1; tin 3 là lương thoả thuận nên không tính.
    assert summary.n == 2
    assert summary.median == pytest.approx(25.0)


def test_salary_summary_with_no_posting_has_n_zero(graph):
    summary = q.salary_summary(graph, q.category_id("acc"), q.province_id("hồ chí minh"))

    assert summary.n == 0 and summary.median is None


def test_top_skills_of_a_title_match_pandas_rates(graph):
    top = q.top_skills(graph, q.title_id("lập trình viên"), k=3)

    assert [s.skill for s in top][:1] == ["python"]
    python = top[0]
    assert python.rate == pytest.approx(1.0) and python.n_with == 3 and python.n_group == 3
    assert set(python.posting_ids) <= {0, 1, 2}


def test_top_skills_of_category_include_chi2(graph):
    top = q.top_skills(graph, q.category_id("acc"), k=5)
    excel = next(s for s in top if s.skill == "excel")

    assert excel.rate == pytest.approx(1.0)
    assert excel.chi2 is not None and excel.chi2 > 0


def test_missing_skills_exclude_what_the_user_has_even_by_alias(graph):
    missing = q.missing_skills(graph, have=["Excel"], group=q.category_id("acc"), k=5)

    names = [s.skill for s in missing]
    assert "excel" not in names
    assert names[0] == "misa"


def test_related_skills_use_cooccurrence(graph):
    related = q.related_skills(graph, "misa", k=5)

    top = related[0]
    assert top.skill == "excel" and top.count == 2


def test_graph_without_resolution_keeps_aliases_as_separate_skills():
    g = build_graph(_postings(), _mentions(), MERGES.iloc[0:0], SKILL_MAP, **SMALL)

    assert g.has_node(q.skill_id("reactjs"))
    assert not g.has_node(q.skill_id("react"))


@pytest.mark.data
def test_real_graph_salary_medians_match_pandas():
    import pickle

    from career_advisor.config import PROCESSED_DIR

    path = PROCESSED_DIR / "graph.pkl"
    if not path.exists():
        pytest.skip("chưa dựng đồ thị")
    with path.open("rb") as f:
        G = pickle.load(f)
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    paid = postings[postings.salary_mid.notna()]
    combos = paid.explode("provinces")[["category", "provinces", "experience_level"]].drop_duplicates()
    # "unknown" và "foreign" không phải tỉnh nên không có node (xem test riêng ở trên).
    combos = combos[~combos.provinces.isin(["unknown", "foreign"])]
    for category, province, level in combos.sample(20, random_state=0).itertuples(index=False):
        in_province = postings.provinces.map(lambda ps, p=province: p in list(ps))
        subset = postings[
            (postings.category == category) & in_province & (postings.experience_level == level)
        ]
        expected = subset.salary_mid.dropna()

        summary = q.salary_summary(
            G, q.category_id(category), q.province_id(province), q.experience_id(level)
        )

        assert summary.n == len(expected)
        assert summary.median == pytest.approx(expected.median())
