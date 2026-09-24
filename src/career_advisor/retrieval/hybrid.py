"""Gộp nhiều cách truy xuất bằng RRF, và bộ xếp hạng dựa trên đồ thị kỹ năng.

RRF (Reciprocal Rank Fusion): điểm của một tin = tổng 1 / (k_rrf + hạng) qua các danh sách.
Chỉ dùng hạng, nên không cần chuẩn hoá điểm giữa TF-IDF, BM25 và cosine.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

import networkx as nx
import pandas as pd

from career_advisor.graph.query import postings_of, resolve_skill, skill_id
from career_advisor.retrieval.lexical import _top_k

K_RRF = 60
DEPTH = 100
MAX_SKILL_WORDS = 5
# Tên kỹ năng quá ngắn dễ khớp nhầm với chữ thường ("ai" nghĩa là "người nào", "c", "r").
MIN_SKILL_CHARS = 3

# Cụm chứa từ nối là danh sách nhiều kỹ năng ("cẩn thận và trung thực"), không phải một kỹ năng.
CONJUNCTIONS = frozenset({"và", "hoặc", "hay"})

_WORD = re.compile(r"\w+")


def rrf(rankings: list[pd.DataFrame], k_rrf: int = K_RRF, k: int = 10) -> pd.DataFrame:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, pid in enumerate(ranking["posting_id"], start=1):
            scores[int(pid)] += 1 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return pd.DataFrame(ordered, columns=["posting_id", "score"])


class HybridSearcher:
    def __init__(self, searchers: list, depth: int = DEPTH, k_rrf: int = K_RRF) -> None:
        self.searchers = searchers
        self.depth = depth
        self.k_rrf = k_rrf

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        return rrf([s.search(query, k=self.depth) for s in self.searchers], self.k_rrf, k)


def extract_skills(G: nx.DiGraph, text: str, max_words: int = MAX_SKILL_WORDS) -> set[str]:
    """Kỹ năng trong đồ thị xuất hiện trong văn bản, qua cả alias.

    Lấy mọi cụm 1 đến `max_words` âm tiết khớp được, rồi bỏ cụm là mảnh của một cụm dài hơn khi tên
    kỹ năng của cụm dài chứa nó ("lập" trong "lập trình"). Cụm dài không được "nuốt" kỹ năng khác:
    alias "python django" gộp vào "django" không làm mất "python" đứng cạnh.
    """
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    matches = []  # (đầu, cuối, cụm, kỹ năng)
    for i in range(len(words)):
        for n in range(1, min(max_words, len(words) - i) + 1):
            if CONJUNCTIONS & set(words[i : i + n]):
                break
            phrase = " ".join(words[i : i + n])
            skill = resolve_skill(G, phrase)
            if skill and len(skill) >= MIN_SKILL_CHARS:
                matches.append((i, i + n, phrase, skill))

    def is_fragment(m: tuple) -> bool:
        start, end, phrase, _ = m
        return any(
            s <= start and end <= e and (e - s) > (end - start) and phrase in skill
            for s, e, _, skill in matches
        )

    return {skill for m in matches if not is_fragment(m) for skill in [m[3]]}


class GraphSearcher:
    """Xếp tin theo kỹ năng trùng với CV trên đồ thị. Kỹ năng hiếm (IDF cao) được tính nặng hơn."""

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G
        self.n_postings = G.graph["n_postings"]

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        scores: dict[int, float] = defaultdict(float)
        for skill in extract_skills(self.G, query):
            holders = postings_of(self.G, skill_id(skill))
            if not holders:
                continue
            idf = math.log(self.n_postings / len(holders))
            for pid in holders:
                scores[pid] += idf
        if not scores:
            return pd.DataFrame(columns=["posting_id", "score"])
        ids = pd.Series(scores)
        return _top_k(ids.index.to_numpy(), ids.to_numpy(), k)
