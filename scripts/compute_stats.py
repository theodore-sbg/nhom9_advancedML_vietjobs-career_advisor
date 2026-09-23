"""Bước stats: multi-hot, χ², đồng xuất hiện và tỷ lệ kỹ năng theo nhóm → data/processed/stats/."""

import pandas as pd

from career_advisor.config import PROCESSED_DIR
from career_advisor.stats import (
    MIN_PAIR_POSTINGS,
    MIN_SKILL_POSTINGS,
    chi2_association,
    cooccurrence,
    group_skill_rates,
    multi_hot,
    resolve_skills,
)

STATS_DIR = PROCESSED_DIR / "stats"
# Kỹ năng từ 5 tin trở lên mới thành node trong đồ thị.
MIN_GRAPH_SKILL_POSTINGS = 5
CHI2_GROUPS = {"category": "category", "experience": "experience_level", "salary": "salary_band"}

if __name__ == "__main__":
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    mentions = resolve_skills(
        pd.read_parquet(PROCESSED_DIR / "skill_mentions.parquet"),
        pd.read_parquet(PROCESSED_DIR / "skill_merges.parquet"),
    )
    mentions.to_parquet(STATS_DIR / "posting_skills.parquet")

    X30, skills30, ids = multi_hot(mentions, postings.index, MIN_SKILL_POSTINGS)
    for name, column in CHI2_GROUPS.items():
        result = chi2_association(X30, skills30, postings.loc[ids, column])
        result.to_parquet(STATS_DIR / f"chi2_{name}.parquet")

    X5, skills5, ids = multi_hot(mentions, postings.index, MIN_GRAPH_SKILL_POSTINGS)
    pairs = cooccurrence(X5, skills5, MIN_PAIR_POSTINGS)
    pairs.to_parquet(STATS_DIR / "cooccurrence.parquet")
    group_skill_rates(X5, skills5, postings.loc[ids, "category"]).to_parquet(
        STATS_DIR / "category_rates.parquet"
    )
    group_skill_rates(X5, skills5, postings.loc[ids, "job_title_node"]).to_parquet(
        STATS_DIR / "title_rates.parquet"
    )

    print(f"Kỹ năng ≥ {MIN_SKILL_POSTINGS} tin (kiểm định χ²): {len(skills30):,}")
    print(f"Kỹ năng ≥ {MIN_GRAPH_SKILL_POSTINGS} tin (node đồ thị): {len(skills5):,}")
    print(f"Cặp đồng xuất hiện ≥ {MIN_PAIR_POSTINGS} tin: {len(pairs):,}")
    chi2 = pd.read_parquet(STATS_DIR / "chi2_category.parquet")
    top = chi2[chi2.positive & chi2.significant]
    for category in ("công_nghệ_thông_tin_kỹ_thuật_số", "tài_chính_kế_toán_ngân_hàng_bảo_hiểm"):
        best = top[top.group == category].nlargest(10, "chi2")["skill"].tolist()
        print(f"Top χ² {category}: {best}")
