"""Bộ nhãn tay cho bước gộp tên kỹ năng: lấy mẫu cặp và lưu nhãn.

Mẫu chia tầng để nhãn không chỉ toàn cặp dễ:
- 5 tầng theo khoảng cosine bge-m3, để đo tầng embedding và tầng LLM ở mọi mức giống nhau;
- 1 tầng "rule": hai cách ghi mà tầng luật đã gộp, để đo precision của chính tầng luật.
Mỗi tầng chia đôi dev/test. Ngưỡng chỉ được chọn trên dev.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_advisor.cleaning.resolve import candidate_pairs
from career_advisor.cleaning.skills import split_compound
from career_advisor.config import SEED

BINS = ((0.78, 0.84), (0.84, 0.88), (0.88, 0.92), (0.92, 0.95), (0.95, 1.0001))
PER_STRATUM = 50
# Chỉ lấy cặp có ít nhất một kỹ năng từ 3 tin trở lên. Nếu không, mẫu gần như toàn chuỗi xuất hiện 1 lần.
MIN_COUNT = 3
LABELS = ("same", "different", "skip")
COLUMNS = ["pair_id", "a", "b", "stratum", "cosine", "split", "label"]


def sample_cosine_pairs(
    skills: pd.DataFrame,
    vectors: np.ndarray,
    bins: tuple[tuple[float, float], ...] = BINS,
    per_bin: int = PER_STRATUM,
    min_count: int = MIN_COUNT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Lấy `per_bin` cặp ngẫu nhiên trong mỗi khoảng cosine. `skills` có cột skill, count."""
    rng = np.random.default_rng(seed)
    pairs = candidate_pairs(vectors, min(low for low, _ in bins))
    counts = skills["count"].to_numpy()
    pairs = pairs[np.maximum(counts[pairs["i"].astype(int)], counts[pairs["j"].astype(int)]) >= min_count]
    names = skills["skill"].to_numpy()
    samples = []
    for k, (low, high) in enumerate(bins):
        in_bin = pairs[(pairs["score"] >= low) & (pairs["score"] < high)]
        picked = in_bin.iloc[np.sort(rng.choice(len(in_bin), size=min(per_bin, len(in_bin)), replace=False))]
        samples.append(
            pd.DataFrame(
                {
                    "a": names[picked["i"].astype(int)],
                    "b": names[picked["j"].astype(int)],
                    "stratum": f"cosine_{k}",
                    "cosine": picked["score"].to_numpy(),
                }
            )
        )
    return pd.concat(samples, ignore_index=True)


def sample_rule_pairs(skill_map: pd.DataFrame, n: int = PER_STRATUM, seed: int = SEED) -> pd.DataFrame:
    """Mỗi kỹ năng lấy tối đa 1 cặp gồm hai cách ghi khác nhau mà tầng luật đã gộp.

    Bỏ chuỗi ghép nhiều kỹ năng như "Tin học văn phòng (Word, Excel)", vì cặp như vậy khó gán nhãn.
    """
    rng = np.random.default_rng(seed)
    single = skill_map[skill_map["raw"].map(lambda r: len(split_compound(r)) == 1)]
    rows = []
    for canonical, group in single.groupby("canonical", sort=True):
        forms = group.drop_duplicates(subset="raw").assign(key=lambda d: d["raw"].str.lower().str.strip())
        forms = forms.drop_duplicates("key")["raw"].tolist()
        if len(forms) >= 2:
            a, b = rng.choice(len(forms), size=2, replace=False)
            rows.append((forms[a], forms[b], canonical))
    pairs = pd.DataFrame(rows, columns=["a", "b", "canonical"])
    if len(pairs) > n:
        pairs = pairs.iloc[np.sort(rng.choice(len(pairs), size=n, replace=False))]
    return pairs[["a", "b"]].assign(stratum="rule", cosine=np.nan).reset_index(drop=True)


def build_label_sheet(pairs: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Bỏ cặp trùng (không kể thứ tự), chia dev/test trong từng tầng, xáo thứ tự để gán nhãn."""
    rng = np.random.default_rng(seed)
    key = [frozenset(p) for p in zip(pairs["a"], pairs["b"], strict=True)]
    pairs = pairs[~pd.Series(key, index=pairs.index).duplicated()].reset_index(drop=True)

    parts = []
    for stratum, group in pairs.groupby("stratum", sort=True):
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        group["split"] = np.where(np.arange(len(group)) % 2 == 0, "dev", "test")
        group["pair_id"] = [f"{stratum}-{k:03d}" for k in range(len(group))]
        parts.append(group)
    sheet = pd.concat(parts, ignore_index=True)
    sheet = sheet.iloc[rng.permutation(len(sheet))].reset_index(drop=True)
    sheet["label"] = ""
    return sheet[COLUMNS]


def load_sheet(path) -> pd.DataFrame:
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet["cosine"] = pd.to_numeric(sheet["cosine"], errors="coerce")
    return sheet[COLUMNS]


def save_sheet(sheet: pd.DataFrame, path) -> None:
    """Ghi qua tệp tạm rồi đổi tên, để tắt ngang giữa chừng cũng không hỏng tệp nhãn."""
    tmp = path.with_suffix(".tmp")
    sheet.to_csv(tmp, index=False)
    tmp.replace(path)


def set_label(sheet: pd.DataFrame, pair_id: str, label: str) -> pd.DataFrame:
    if label not in LABELS:
        raise ValueError(f"Nhãn phải là một trong {LABELS}, nhận {label!r}")
    out = sheet.copy()
    out.loc[out["pair_id"] == pair_id, "label"] = label
    return out


def next_unlabeled(sheet: pd.DataFrame) -> int | None:
    """Vị trí (theo thứ tự dòng) của cặp đầu tiên chưa gán nhãn."""
    empty = np.flatnonzero(sheet["label"].to_numpy() == "")
    return int(empty[0]) if len(empty) else None
