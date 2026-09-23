"""Thống kê kỹ năng: multi-hot, kiểm định χ², đồng xuất hiện (PMI) và tỷ lệ theo nhóm.

Mọi con số ở đây được ghi lên cạnh của đồ thị. LLM không tự tính số nào.
Các phép tính dùng ma trận thưa tin × kỹ năng, nên chạy trên cả 48 nghìn tin trong vài giây.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist

# Chỉ kiểm định kỹ năng có từ 30 tin trở lên. Kỹ năng hiếm cho bảng 2×2 có ô kỳ vọng quá nhỏ.
MIN_SKILL_POSTINGS = 30
MIN_PAIR_POSTINGS = 5
ALPHA = 0.05


def resolve_skills(mentions: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên kỹ năng theo bảng gộp (skill → canonical), rồi bỏ trùng trong cùng tin."""
    rename = dict(zip(merges["skill"], merges["canonical"], strict=True))
    out = mentions.assign(skill=mentions["skill"].map(lambda s: rename.get(s, s)))
    return out.drop_duplicates(["posting_id", "skill"]).reset_index(drop=True)


def multi_hot(
    mentions: pd.DataFrame, posting_ids: pd.Index, min_count: int = MIN_SKILL_POSTINGS
) -> tuple[sparse.csr_matrix, list[str], pd.Index]:
    """Ma trận thưa tin × kỹ năng (0/1). Hàng theo `posting_ids`, kể cả tin không có kỹ năng nào."""
    counts = mentions.groupby("skill")["posting_id"].nunique()
    skills = sorted(counts[counts >= min_count].index)
    kept = mentions[mentions["skill"].isin(skills)].drop_duplicates(["posting_id", "skill"])
    row_of = pd.Series(np.arange(len(posting_ids)), index=posting_ids)
    col_of = pd.Series(np.arange(len(skills)), index=skills)
    rows = row_of.loc[kept["posting_id"]].to_numpy()
    cols = col_of.loc[kept["skill"]].to_numpy()
    data = np.ones(len(kept), dtype=np.int32)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(posting_ids), len(skills)))
    return X, skills, posting_ids


def _group_counts(X: sparse.csr_matrix, groups: pd.Series):
    """Bỏ tin không có nhóm, rồi đếm số tin có kỹ năng trong từng nhóm."""
    mask = groups.notna().to_numpy()
    Xg = X[mask]
    codes, names = pd.factorize(groups[mask], sort=True)
    G = sparse.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(names))
    )
    with_skill = np.asarray((G.T @ Xg).todense())  # nhóm × kỹ năng
    n_group = np.asarray(G.sum(axis=0)).ravel()
    n_skill = np.asarray(Xg.sum(axis=0)).ravel()
    return with_skill, n_group, n_skill, len(codes), list(names)


def chi2_association(
    X: sparse.csr_matrix, skills: list[str], groups: pd.Series, alpha: float = ALPHA
) -> pd.DataFrame:
    """Kiểm định χ² (bảng 2×2, không hiệu chỉnh Yates) cho mỗi cặp kỹ năng × nhóm.

    `groups` cùng thứ tự hàng với `X`. Tin có nhóm thiếu (NaN) bị bỏ.
    `significant` dùng hiệu chỉnh Bonferroni theo tổng số phép kiểm định.
    `positive` là True khi kỹ năng phổ biến trong nhóm hơn ngoài nhóm.
    """
    a, n_group, n_skill, n, names = _group_counts(X, groups)
    n_g = n_group[:, None].astype(float)
    n_s = n_skill[None, :].astype(float)
    b, c = n_g - a, n_s - a
    d = n - n_g - n_s + a
    denom = n_g * (n - n_g) * n_s * (n - n_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(denom > 0, n * (a * d - b * c) ** 2 / denom, 0.0)
        rate_out = np.where(n - n_g > 0, c / (n - n_g), 0.0)
    rate_in = a / n_g
    p = chi2_dist.sf(chi2, df=1)

    result = pd.DataFrame(
        {
            "group": np.repeat(names, len(skills)),
            "skill": np.tile(skills, len(names)),
            "chi2": chi2.ravel(),
            "p_value": p.ravel(),
            "n_with_skill_in_group": a.ravel().astype(int),
            "n_group": np.repeat(n_group, len(skills)).astype(int),
            "rate_in_group": rate_in.ravel(),
            "rate_outside": rate_out.ravel(),
        }
    )
    result["positive"] = result["rate_in_group"] > result["rate_outside"]
    result["significant"] = result["p_value"] < alpha / len(result)
    return result


def cooccurrence(X: sparse.csr_matrix, skills: list[str], min_pair: int = MIN_PAIR_POSTINGS) -> pd.DataFrame:
    """Cặp kỹ năng cùng xuất hiện trong ít nhất `min_pair` tin, kèm PMI và PMI chuẩn hoá (NPMI).

    PMI = log(p(a,b) / (p(a) p(b))). NPMI chia PMI cho -log p(a,b), nằm trong [-1, 1].
    """
    n = X.shape[0]
    C = sparse.triu(X.T @ X, k=1).tocoo()
    keep = C.data >= min_pair
    i, j, both = C.row[keep], C.col[keep], C.data[keep].astype(float)
    per_skill = np.asarray(X.sum(axis=0)).ravel().astype(float)
    p_ab = both / n
    pmi = np.log(p_ab / ((per_skill[i] / n) * (per_skill[j] / n)))
    names = np.array(skills)
    return pd.DataFrame(
        {
            "skill_a": names[i],
            "skill_b": names[j],
            "count": both.astype(int),
            "pmi": pmi,
            "npmi": pmi / -np.log(p_ab),
        }
    ).sort_values(["skill_a", "skill_b"], ignore_index=True)


def group_skill_rates(
    X: sparse.csr_matrix, skills: list[str], groups: pd.Series, min_with: int = 3
) -> pd.DataFrame:
    """Tỷ lệ tin trong mỗi nhóm (chức danh, nhóm ngành) có yêu cầu từng kỹ năng."""
    a, n_group, _, _, names = _group_counts(X, groups)
    g, s = np.nonzero(a >= min_with)
    return pd.DataFrame(
        {
            "group": np.array(names)[g],
            "skill": np.array(skills)[s],
            "n_with": a[g, s].astype(int),
            "n_group": n_group[g].astype(int),
            "rate": a[g, s] / n_group[g],
        }
    ).sort_values(["group", "rate"], ascending=[True, False], ignore_index=True)
