import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from career_advisor.stats import chi2_association, cooccurrence, group_skill_rates, multi_hot, resolve_skills


def _mentions(rows: dict[int, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(pid, s) for pid, skills in rows.items() for s in skills], columns=["posting_id", "skill"]
    )


# 8 tin: 4 tin CNTT, 4 tin kế toán.
MENTIONS = _mentions(
    {
        0: ["python", "git"],
        1: ["python", "git", "excel"],
        2: ["python"],
        3: ["git"],
        4: ["excel", "misa"],
        5: ["excel", "misa"],
        6: ["excel"],
        7: ["excel", "python"],
    }
)
CATEGORY = pd.Series(["it"] * 4 + ["acc"] * 4, index=pd.RangeIndex(8, name="posting_id"))


def test_resolve_skills_applies_merges_and_dedupes_within_posting():
    mentions = _mentions({0: ["reactjs", "react", "git"], 1: ["reactjs"]})
    merges = pd.DataFrame({"skill": ["reactjs"], "canonical": ["react"]})

    out = resolve_skills(mentions, merges)

    assert sorted(out[out.posting_id == 0].skill) == ["git", "react"]
    assert out[out.posting_id == 1].skill.tolist() == ["react"]


def test_multi_hot_keeps_skills_with_enough_postings_and_is_binary():
    X, skills, postings = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=2)

    assert skills == ["excel", "git", "misa", "python"]
    assert X.shape == (8, 4)
    assert set(np.unique(X.toarray())) <= {0, 1}
    assert X[:, skills.index("excel")].sum() == 5
    assert list(postings) == list(range(8))


def test_chi2_matches_scipy_on_the_2x2_table():
    X, skills, postings = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=1)

    result = chi2_association(X, skills, CATEGORY.loc[postings])
    row = result[(result.skill == "python") & (result.group == "it")].iloc[0]

    # python: 3/4 tin CNTT, 1/4 tin kế toán.
    table = np.array([[3, 1], [1, 3]])
    expected_chi2, expected_p, _, _ = chi2_contingency(table, correction=False)
    assert row.chi2 == pytest.approx(expected_chi2)
    assert row.p_value == pytest.approx(expected_p)
    assert row.n_with_skill_in_group == 3
    assert row.rate_in_group == pytest.approx(0.75)
    assert row.rate_outside == pytest.approx(0.25)
    assert row.positive


def test_chi2_ignores_postings_without_a_group():
    X, skills, postings = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=1)
    groups = CATEGORY.loc[postings].copy()
    groups.iloc[7] = None  # tin lương thoả thuận chẳng hạn

    result = chi2_association(X, skills, groups)
    row = result[(result.skill == "python") & (result.group == "it")].iloc[0]

    table = np.array([[3, 1], [0, 3]])  # tin 7 (acc, có python) bị bỏ
    assert row.chi2 == pytest.approx(chi2_contingency(table, correction=False)[0])


def test_chi2_flags_bonferroni_significance():
    X, skills, postings = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=1)

    result = chi2_association(X, skills, CATEGORY.loc[postings], alpha=0.05)

    n_tests = len(result)
    assert (result.significant == (result.p_value < 0.05 / n_tests)).all()


def test_cooccurrence_counts_and_pmi_match_pandas():
    X, skills, _ = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=1)

    pairs = cooccurrence(X, skills, min_pair=2)
    row = pairs[(pairs.skill_a == "excel") & (pairs.skill_b == "misa")].iloc[0]

    has = MENTIONS.assign(v=1).pivot_table(index="posting_id", columns="skill", values="v", fill_value=0)
    n = len(has)
    both = int(((has.excel == 1) & (has.misa == 1)).sum())
    p_ab, p_a, p_b = both / n, has.excel.mean(), has.misa.mean()
    assert row["count"] == both == 2
    assert row.pmi == pytest.approx(math.log(p_ab / (p_a * p_b)))
    assert row.npmi == pytest.approx(math.log(p_ab / (p_a * p_b)) / -math.log(p_ab))
    assert (pairs.skill_a < pairs.skill_b).all()
    assert (pairs["count"] >= 2).all()


def test_group_skill_rates_match_pandas_groupby():
    X, skills, postings = multi_hot(MENTIONS, posting_ids=CATEGORY.index, min_count=1)

    rates = group_skill_rates(X, skills, CATEGORY.loc[postings], min_with=1)
    row = rates[(rates.group == "acc") & (rates.skill == "excel")].iloc[0]

    assert row.n_with == 4 and row.n_group == 4 and row.rate == 1.0
    it_git = rates[(rates.group == "it") & (rates.skill == "git")].iloc[0]
    assert it_git.rate == pytest.approx(3 / 4)
