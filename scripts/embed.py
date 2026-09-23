"""Mã hoá kỹ năng và tin tuyển dụng bằng bge-m3.

python scripts/embed.py skills      # nhanh (khoảng 2 phút): kỹ năng sau tầng luật
python scripts/embed.py postings    # lâu (khoảng 60 phút): 48 nghìn tin, nên chạy nền

Gộp tên kỹ năng từ các vector này nằm ở scripts/resolve_skills.py.
"""

import sys

import numpy as np
import pandas as pd

from career_advisor.config import PROCESSED_DIR
from career_advisor.embeddings import BgeM3Encoder, encode_in_chunks

EMB_DIR = PROCESSED_DIR / "emb"


def skill_table() -> pd.DataFrame:
    """Mỗi kỹ năng (sau tầng luật) một dòng: số tin nhắc tới và loại."""
    mentions = pd.read_parquet(PROCESSED_DIR / "skill_mentions.parquet")
    kinds = pd.read_parquet(PROCESSED_DIR / "skill_map.parquet").drop_duplicates("canonical")
    counts = mentions.groupby("skill").size().rename("count").reset_index()
    table = counts.merge(kinds[["canonical", "kind"]], left_on="skill", right_on="canonical")
    return table.drop(columns="canonical").sort_values("skill").reset_index(drop=True)


def posting_texts() -> list[str]:
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    parts = postings[["job_title", "requirements_text", "description"]].fillna("")
    return [f"{t}. {r}. {d}" for t, r, d in parts.itertuples(index=False)]


def run_skills(encoder: BgeM3Encoder) -> None:
    skills = skill_table()
    skills.to_parquet(EMB_DIR / "skills_index.parquet")
    vectors = encode_in_chunks(skills["skill"].tolist(), EMB_DIR / "skills", encoder)
    print(f"Đã mã hoá {len(vectors):,} kỹ năng")


def run_postings(encoder: BgeM3Encoder) -> None:
    vectors = encode_in_chunks(posting_texts(), EMB_DIR / "postings", encoder)
    np.save(EMB_DIR / "postings.npy", vectors)
    print(f"Đã mã hoá {len(vectors):,} tin, kích thước {vectors.shape[1]}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["skills", "postings"]
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    encoder = BgeM3Encoder()
    if "skills" in targets:
        run_skills(encoder)
    if "postings" in targets:
        run_postings(encoder)
