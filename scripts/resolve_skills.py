"""Gộp tên kỹ năng theo tầng embedding và tầng LLM → data/processed/skill_merges.parquet.

Cần chạy `scripts/embed.py skills` trước. Ghi thêm skill_merges_embedding.parquet (chỉ tầng
embedding) để đánh giá riêng từng tầng.
"""

import sys

import numpy as np
import pandas as pd

from career_advisor.cleaning.resolve import (
    AUTO_THRESHOLD,
    LLM_BATCH,
    LLM_LOW,
    judge_pairs,
    resolve_tiers,
)
from career_advisor.config import PROCESSED_DIR
from career_advisor.llm import make_client

MAX_CALLS = 500  # quá mức này thì dừng lại hỏi người dùng (SPEC mục Ranh giới)


def load_skills() -> tuple[pd.DataFrame, np.ndarray]:
    skills = pd.read_parquet(PROCESSED_DIR / "emb" / "skills_index.parquet")
    chunks = sorted((PROCESSED_DIR / "emb" / "skills").glob("0*.npy"))
    return skills, np.concatenate([np.load(c) for c in chunks])


if __name__ == "__main__":
    skills, vectors = load_skills()
    embedding_only = resolve_tiers(skills, vectors, AUTO_THRESHOLD, judge=None)
    embedding_only.to_parquet(PROCESSED_DIR / "skill_merges_embedding.parquet")

    client = make_client()

    def judge(pairs: list[tuple[str, str]]) -> list[bool]:
        calls = -(-len(pairs) // LLM_BATCH)
        print(f"Tầng LLM: {len(pairs):,} cặp đại diện → tối đa {calls} lượt gọi {client.model}")
        if calls > MAX_CALLS and "--yes" not in sys.argv:
            sys.exit(f"Vượt {MAX_CALLS} lượt. Chạy lại với --yes nếu người dùng đồng ý.")
        return judge_pairs(client, pairs)

    merges = resolve_tiers(skills, vectors, AUTO_THRESHOLD, LLM_LOW, judge=judge)
    merges.to_parquet(PROCESSED_DIR / "skill_merges.parquet")
    by_tier = merges["tier"].value_counts().to_dict()
    print(f"Ngưỡng: tự gộp ≥ {AUTO_THRESHOLD}, hỏi LLM [{LLM_LOW}, {AUTO_THRESHOLD})")
    print(f"Lượt gọi thật: {client.calls} (phần còn lại lấy từ cache)")
    print(f"Gộp {len(merges):,} tên {by_tier} → còn {len(skills) - len(merges):,} kỹ năng")
