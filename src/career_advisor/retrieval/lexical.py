"""Truy xuất từ vựng: TF-IDF và BM25 trên văn bản tin tuyển dụng.

Từ tiếng Việt thường gồm nhiều âm tiết ("kế toán", "phần mềm"), nên mỗi văn bản được tách thành
âm tiết cộng cặp âm tiết liền nhau ("kế_toán"). CV và tin đi qua cùng một hàm tách.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    syllables = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    return syllables + [f"{a}_{b}" for a, b in zip(syllables, syllables[1:], strict=False)]


def posting_documents(postings: pd.DataFrame, skills: pd.DataFrame) -> pd.Series:
    """Văn bản đại diện mỗi tin: chức danh, kỹ năng đã chuẩn hoá, yêu cầu, mô tả."""
    skill_text = skills.groupby("posting_id")["skill"].apply(lambda s: ", ".join(sorted(s)))
    parts = postings[["job_title", "requirements_text", "description"]].fillna("")
    skill_text = skill_text.reindex(postings.index).fillna("")
    return pd.Series(
        [
            ". ".join(x for x in (t, s, r, d) if x)
            for t, s, r, d in zip(
                parts["job_title"], skill_text, parts["requirements_text"], parts["description"], strict=True
            )
        ],
        index=postings.index,
    )


def _top_k(ids: np.ndarray, scores: np.ndarray, k: int) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")[:k]
    order = order[scores[order] > 0]
    return pd.DataFrame({"posting_id": ids[order], "score": scores[order].astype(float)})


class TfidfSearcher:
    def __init__(self, docs: pd.Series) -> None:
        self.ids = docs.index.to_numpy()
        self.vectorizer = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs.tolist())

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return _top_k(self.ids, scores, k)


class Bm25Searcher:
    def __init__(self, docs: pd.Series) -> None:
        self.ids = docs.index.to_numpy()
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        return _top_k(self.ids, scores, k)
