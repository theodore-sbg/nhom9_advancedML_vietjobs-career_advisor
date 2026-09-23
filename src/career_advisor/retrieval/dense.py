"""Truy xuất ngữ nghĩa bằng embedding bge-m3 đã mã hoá sẵn cho 48 nghìn tin (Task 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_advisor.embeddings import Encoder
from career_advisor.retrieval.lexical import _top_k


class DenseSearcher:
    """`vectors[i]` là embedding (độ dài 1) của tin `ids[i]`. Điểm là cosine với CV."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, encoder: Encoder) -> None:
        self.vectors = vectors
        self.ids = np.asarray(ids)
        self.encoder = encoder

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        scores = self.vectors @ self.encoder.encode([query])[0]
        return _top_k(self.ids, scores, k)
