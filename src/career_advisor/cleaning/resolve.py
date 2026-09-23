"""Tầng 2 của bước gộp tên kỹ năng: gộp theo độ giống nhau của embedding bge-m3.

Cặp có cosine ≥ ngưỡng cao được gộp tự động. Cặp ở vùng không chắc chắn để tầng LLM (Task 8) quyết.
Gộp bằng union-find, xét cặp giống nhất trước, và có 3 chốt chặn để tránh gộp nhầm:
- cụm không vượt quá `max_cluster` thành viên, để chuỗi A≈B≈C không kéo hai kỹ năng khác xa vào nhau;
- cặp cấm gộp (java và javascript…) không được nằm chung cụm, kể cả gián tiếp;
- hai tên khác số ("2d", "3d") hoặc khác ngôn ngữ ("tiếng anh", "tiếng trung") không gộp.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Ngưỡng tạm. Task 8 chọn lại trên phần dev của bộ nhãn tay.
AUTO_THRESHOLD = 0.92
MAX_CLUSTER = 30

# Các cặp embedding hay thấy giống nhưng là kỹ năng khác nhau.
FORBIDDEN: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("java", "javascript"), ("c", "c++"), ("c", "c#"), ("c++", "c#"), ("react", "react native"),
        ("sql", "nosql"), ("word", "excel"), ("word", "powerpoint"), ("excel", "powerpoint"),
        ("photoshop", "illustrator"), ("premiere", "after effects"), ("autocad", "revit"),
        ("ios", "android"), ("frontend", "backend"), ("front end", "back end"),
    )
)  # fmt: skip

_DIGITS = re.compile(r"\d+")
_LANGUAGE = re.compile(r"tiếng (\w+)")


def candidate_pairs(vectors: np.ndarray, threshold: float, block: int = 1024) -> pd.DataFrame:
    """Các cặp (i, j), i < j, có cosine ≥ `threshold`. `vectors` phải đã chuẩn hoá độ dài 1.

    Tính theo khối hàng để không phải giữ cả ma trận n × n trong bộ nhớ.
    """
    n = len(vectors)
    cols = np.arange(n)
    found = []
    for start in range(0, n, block):
        rows = np.arange(start, min(start + block, n))
        sims = vectors[rows] @ vectors.T
        mask = (sims >= threshold) & (cols[None, :] > rows[:, None])
        r, c = np.nonzero(mask)
        found.append(pd.DataFrame({"i": rows[r], "j": c, "score": sims[r, c].astype(float)}))
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["i", "j", "score"])


def _conflict(a: str, b: str) -> bool:
    if frozenset((a, b)) in FORBIDDEN:
        return True
    digits_a, digits_b = set(_DIGITS.findall(a)), set(_DIGITS.findall(b))
    if digits_a and digits_b and digits_a != digits_b:
        return True
    lang_a, lang_b = set(_LANGUAGE.findall(a)), set(_LANGUAGE.findall(b))
    return bool(lang_a and lang_b and lang_a != lang_b)


def merge_clusters(
    skills: list[str], counts: list[int], pairs: pd.DataFrame, max_cluster: int = MAX_CLUSTER
) -> dict[str, str]:
    """Gộp các cặp thành cụm, trả `skill → đại diện`. Đại diện là thành viên xuất hiện nhiều nhất."""
    parent = list(range(len(skills)))
    members = {i: [i] for i in range(len(skills))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ordered = pairs.sort_values(["score", "i", "j"], ascending=[False, True, True])
    for i, j in zip(ordered["i"].astype(int), ordered["j"].astype(int), strict=True):
        ri, rj = find(i), find(j)
        if ri == rj or len(members[ri]) + len(members[rj]) > max_cluster:
            continue
        if any(_conflict(skills[a], skills[b]) for a in members[ri] for b in members[rj]):
            continue
        parent[rj] = ri
        members[ri].extend(members.pop(rj))

    rep = {}
    for group in members.values():
        best = min(group, key=lambda k: (-counts[k], len(skills[k]), skills[k]))
        rep.update({skills[k]: skills[best] for k in group})
    return rep


def embedding_tier(
    skills: pd.DataFrame,
    vectors: np.ndarray,
    threshold: float = AUTO_THRESHOLD,
    max_cluster: int = MAX_CLUSTER,
) -> pd.DataFrame:
    """Gộp tự động các kỹ năng cùng loại có cosine ≥ `threshold`.

    `skills` có cột skill, count, kind và cùng thứ tự với `vectors`.
    Trả các dòng bị gộp: skill, canonical, tier="embedding", score (cosine với đại diện).
    """
    pairs = candidate_pairs(vectors, threshold)
    kinds = skills["kind"].to_numpy()
    pairs = pairs[kinds[pairs["i"].astype(int)] == kinds[pairs["j"].astype(int)]]
    names = skills["skill"].tolist()
    rep = merge_clusters(names, skills["count"].tolist(), pairs, max_cluster)

    index = {name: k for k, name in enumerate(names)}
    rows = [
        (s, r, "embedding", float(vectors[index[s]] @ vectors[index[r]])) for s, r in rep.items() if s != r
    ]
    merges = pd.DataFrame(rows, columns=["skill", "canonical", "tier", "score"])
    return merges.sort_values(["canonical", "skill"]).reset_index(drop=True)
