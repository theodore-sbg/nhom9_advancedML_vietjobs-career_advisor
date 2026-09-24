"""Nạp dữ liệu và dựng đủ 5 cách truy xuất. Dùng chung cho đánh giá, agent và giao diện."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from career_advisor.config import PROCESSED_DIR
from career_advisor.retrieval.dense import DenseSearcher
from career_advisor.retrieval.hybrid import GraphSearcher, HybridSearcher
from career_advisor.retrieval.lexical import Bm25Searcher, TfidfSearcher, posting_documents

SYSTEMS = ("tfidf", "bm25", "dense", "hybrid", "hybrid+graph")


def load_resources() -> dict:
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    skills = pd.read_parquet(PROCESSED_DIR / "stats" / "posting_skills.parquet")[["posting_id", "skill"]]
    with (PROCESSED_DIR / "graph.pkl").open("rb") as f:
        graph = pickle.load(f)
    return {"postings": postings, "skills": skills, "graph": graph}


def build_searchers(resources: dict, encoder=None) -> dict:
    """`encoder` mặc định là bge-m3 (nạp mất khoảng 15 giây)."""
    from career_advisor.embeddings import BgeM3Encoder

    postings = resources["postings"]
    docs = posting_documents(postings, resources["skills"])
    tfidf, bm25 = TfidfSearcher(docs), Bm25Searcher(docs)
    dense = DenseSearcher(
        np.load(PROCESSED_DIR / "emb" / "postings.npy"), postings.index.to_numpy(), encoder or BgeM3Encoder()
    )
    graph = GraphSearcher(resources["graph"])
    return {
        "tfidf": tfidf,
        "bm25": bm25,
        "dense": dense,
        "hybrid": HybridSearcher([tfidf, bm25, dense]),
        "hybrid+graph": HybridSearcher([tfidf, bm25, dense, graph]),
    }
