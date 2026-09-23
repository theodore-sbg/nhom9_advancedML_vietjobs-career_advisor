import numpy as np
import pandas as pd
import pytest

from career_advisor.evaluation.skill_pairs import (
    build_label_sheet,
    load_sheet,
    next_unlabeled,
    sample_cosine_pairs,
    sample_rule_pairs,
    save_sheet,
    set_label,
)


def _skills(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(4, 16))
    # 4 nhóm vector gần nhau, để có đủ cặp ở nhiều mức cosine.
    v = base[np.arange(n) % 4] + rng.normal(scale=np.linspace(0.05, 0.9, n))[:, None] * rng.normal(
        size=(n, 16)
    )
    v = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    skills = pd.DataFrame(
        {"skill": [f"kỹ năng {k}" for k in range(n)], "count": [1 + k % 5 for k in range(n)]}
    )
    return skills, v


BINS = ((0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))


def test_cosine_pairs_fall_in_their_bin_and_respect_min_count():
    skills, v = _skills()

    pairs = sample_cosine_pairs(skills, v, bins=BINS, per_bin=5, min_count=3, seed=1)

    counts = dict(zip(skills.skill, skills["count"], strict=True))
    for row in pairs.itertuples():
        low, high = BINS[int(row.stratum.split("_")[1])]
        assert low <= row.cosine < high
        assert max(counts[row.a], counts[row.b]) >= 3


def test_cosine_pairs_are_deterministic_for_a_seed():
    skills, v = _skills()

    first = sample_cosine_pairs(skills, v, bins=BINS, per_bin=5, seed=7)
    second = sample_cosine_pairs(skills, v, bins=BINS, per_bin=5, seed=7)

    assert first.equals(second)


def test_rule_pairs_are_different_surface_forms_of_one_skill():
    skill_map = pd.DataFrame(
        {
            "raw": [
                "ReactJS",
                "React.js",
                "react.js",
                "Excel",
                "MS Excel",
                "Tin học văn phòng (Word, Excel)",
            ],
            "canonical": ["react", "react", "react", "excel", "excel", "excel"],
        }
    )

    pairs = sample_rule_pairs(skill_map, n=10, seed=0)

    assert set(pairs.stratum) == {"rule"}
    for row in pairs.itertuples():
        assert row.a.lower() != row.b.lower()
        assert "(" not in row.a + row.b  # bỏ chuỗi ghép nhiều kỹ năng
    assert len(pairs) == 2  # react: ReactJS–React.js; excel: Excel–MS Excel


def test_label_sheet_has_no_duplicate_pairs_and_balanced_splits():
    pairs = pd.DataFrame(
        {
            "a": ["x", "y", "p", "q", "m", "n"],
            "b": ["y", "x", "q", "r", "n", "o"],
            "stratum": ["s0", "s0", "s0", "s0", "s1", "s1"],
            "cosine": [0.9] * 6,
        }
    )

    sheet = build_label_sheet(pairs, seed=0)

    unordered = {frozenset((a, b)) for a, b in zip(sheet.a, sheet.b, strict=True)}
    assert len(unordered) == len(sheet) == 5
    assert sheet.pair_id.is_unique
    assert set(sheet.split) == {"dev", "test"}
    assert (sheet.label == "").all()
    for _, group in sheet.groupby("stratum"):
        assert abs((group.split == "dev").sum() - (group.split == "test").sum()) <= 1


def test_labels_are_saved_and_resume_at_first_unlabeled(tmp_path):
    pairs = pd.DataFrame({"a": ["a", "c"], "b": ["b", "d"], "stratum": ["s0", "s0"], "cosine": [0.9, 0.8]})
    path = tmp_path / "pairs.csv"
    sheet = build_label_sheet(pairs, seed=0)
    save_sheet(sheet, path)

    sheet = set_label(load_sheet(path), sheet.pair_id.iloc[0], "same")
    save_sheet(sheet, path)
    reloaded = load_sheet(path)

    assert reloaded.label.iloc[0] == "same"
    assert next_unlabeled(reloaded) == 1


def test_all_labeled_returns_none():
    pairs = pd.DataFrame({"a": ["a"], "b": ["b"], "stratum": ["s0"], "cosine": [0.9]})
    sheet = build_label_sheet(pairs, seed=0)

    sheet = set_label(sheet, sheet.pair_id.iloc[0], "different")

    assert next_unlabeled(sheet) is None


def test_invalid_label_is_rejected():
    pairs = pd.DataFrame({"a": ["a"], "b": ["b"], "stratum": ["s0"], "cosine": [0.9]})
    sheet = build_label_sheet(pairs, seed=0)

    with pytest.raises(ValueError):
        set_label(sheet, sheet.pair_id.iloc[0], "maybe")
