"""Chia train/val/test dùng chung cho mọi module.

Bài báo VietJobs không công bố tập chia, nên nhóm tự chia 80/10/10, phân tầng theo nhóm ngành.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_advisor.config import SEED

SPLITS = ("train", "val", "test")
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def make_splits(postings: pd.DataFrame, seed: int = SEED, by: str = "category") -> pd.Series:
    """Trả về Series `posting_id → split`, cùng index với `postings`.

    Mỗi nhóm ngành được xáo trộn và chia riêng, nên tỷ lệ nhóm ngành giữ nguyên trong 3 tập.
    Kết quả không phụ thuộc thứ tự dòng đầu vào.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series(index=postings.index, dtype=pd.CategoricalDtype(SPLITS), name="split")
    for _, group in postings.groupby(by, sort=True):
        ids = rng.permutation(np.sort(group.index.to_numpy()))
        n_train = round(len(ids) * TRAIN_FRAC)
        n_val = round(len(ids) * VAL_FRAC)
        split.loc[ids[:n_train]] = "train"
        split.loc[ids[n_train : n_train + n_val]] = "val"
        split.loc[ids[n_train + n_val :]] = "test"
    return split
