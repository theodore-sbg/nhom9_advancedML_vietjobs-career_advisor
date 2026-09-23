"""Bước knowledge-graph: dựng đồ thị → data/processed/graph.pkl.

python scripts/build_graph.py                  # đồ thị đầy đủ
python scripts/build_graph.py --no-resolution  # không gộp tên kỹ năng (ablation) → graph_no_resolution.pkl
"""

import pickle
import sys
import time
from collections import Counter

import pandas as pd

from career_advisor.config import PROCESSED_DIR
from career_advisor.graph.build import build_graph


def load_inputs(no_resolution: bool):
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    mentions = pd.read_parquet(PROCESSED_DIR / "skill_mentions.parquet")
    merges = pd.read_parquet(PROCESSED_DIR / "skill_merges.parquet")
    if no_resolution:
        merges = merges.iloc[0:0]
    skill_map = pd.read_parquet(PROCESSED_DIR / "skill_map.parquet")
    return postings, mentions, merges, skill_map


if __name__ == "__main__":
    no_resolution = "--no-resolution" in sys.argv
    start = time.time()
    G = build_graph(*load_inputs(no_resolution))
    out = PROCESSED_DIR / ("graph_no_resolution.pkl" if no_resolution else "graph.pkl")
    with out.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dựng xong trong {time.time() - start:.0f} giây → {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    print("Node:", dict(Counter(d["type"] for _, d in G.nodes(data=True)).most_common()))
    print("Cạnh:", dict(Counter(d["rel"] for _, _, d in G.edges(data=True)).most_common()))
