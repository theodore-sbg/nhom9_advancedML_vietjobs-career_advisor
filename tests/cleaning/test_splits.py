import pandas as pd
import pytest

from career_advisor import data
from career_advisor.cleaning.splits import SPLITS, make_splits


def _postings(sizes: dict[str, int]) -> pd.DataFrame:
    categories = [c for c, n in sizes.items() for _ in range(n)]
    df = pd.DataFrame({"category": categories})
    df.index.name = "posting_id"
    return df


def _max_share_gap(df: pd.DataFrame, splits: pd.Series) -> float:
    """Độ lệch lớn nhất (điểm %) giữa tỷ lệ một nhóm ngành trong các tập."""
    shares = pd.crosstab(df["category"], splits, normalize="columns") * 100
    return float((shares.max(axis=1) - shares.min(axis=1)).max())


def test_every_posting_gets_exactly_one_split():
    df = _postings({"a": 100, "b": 37})

    splits = make_splits(df)

    assert splits.index.equals(df.index)
    assert set(splits.unique()) == set(SPLITS)


def test_split_sizes_are_close_to_80_10_10():
    df = _postings({"a": 500, "b": 300, "c": 200})

    counts = make_splits(df).value_counts()

    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100


def test_category_shares_differ_by_at_most_1_point_across_splits():
    df = _postings({"a": 8276, "b": 1906, "c": 384, "d": 322})

    assert _max_share_gap(df, make_splits(df)) <= 1.0


def test_same_seed_gives_same_split():
    df = _postings({"a": 200, "b": 50})

    assert make_splits(df).equals(make_splits(df))


def test_different_seed_gives_different_split():
    df = _postings({"a": 200, "b": 50})

    assert not make_splits(df, seed=1).equals(make_splits(df, seed=2))


def test_split_does_not_depend_on_row_order():
    df = _postings({"a": 200, "b": 50})

    shuffled = df.sample(frac=1, random_state=0)

    assert make_splits(shuffled).sort_index().equals(make_splits(df))


@pytest.mark.data
@pytest.mark.skipif(not data.RAW_CSV.exists(), reason="chưa tải dữ liệu")
def test_real_split_is_stratified():
    df = data.load_postings()

    assert _max_share_gap(df, make_splits(df)) <= 1.0


def test_split_survives_parquet_roundtrip(tmp_path):
    df = _postings({"a": 30, "b": 20})
    splits = make_splits(df)

    splits.to_frame().to_parquet(tmp_path / "s.parquet")

    assert pd.read_parquet(tmp_path / "s.parquet")["split"].equals(splits)
