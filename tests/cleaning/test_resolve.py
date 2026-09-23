import numpy as np
import pandas as pd

from career_advisor.cleaning.resolve import candidate_pairs, embedding_tier, merge_clusters


def _unit(*rows: list[float]) -> np.ndarray:
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _pairs(*rows: tuple[int, int, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["i", "j", "score"])


def test_candidate_pairs_keeps_only_pairs_above_threshold():
    vectors = _unit([1, 0], [0.99, 0.14], [0, 1])

    pairs = candidate_pairs(vectors, threshold=0.9)

    assert pairs[["i", "j"]].values.tolist() == [[0, 1]]
    assert abs(pairs.score.iloc[0] - float(vectors[0] @ vectors[1])) < 1e-6


def test_candidate_pairs_is_the_same_across_block_sizes():
    rng = np.random.default_rng(0)
    vectors = _unit(*rng.normal(size=(50, 8)).tolist())

    small = candidate_pairs(vectors, threshold=0.5, block=7).sort_values(["i", "j"]).reset_index(drop=True)
    large = candidate_pairs(vectors, threshold=0.5, block=1000).sort_values(["i", "j"]).reset_index(drop=True)

    assert small[["i", "j"]].equals(large[["i", "j"]])
    # float32 nhân ma trận theo khối khác nhau lệch ở chữ số thứ 7.
    assert np.allclose(small.score, large.score, atol=1e-6)
    assert (small.i < small.j).all()


def test_chain_merges_into_most_frequent_member():
    skills = ["reactjs", "react", "react.js"]

    rep = merge_clusters(skills, counts=[5, 50, 3], pairs=_pairs((0, 1, 0.97), (1, 2, 0.96)))

    assert rep == {"reactjs": "react", "react": "react", "react.js": "react"}


def test_cluster_size_is_capped():
    skills = ["a", "b", "c", "d"]
    pairs = _pairs((0, 1, 0.99), (1, 2, 0.98), (2, 3, 0.97))

    rep = merge_clusters(skills, counts=[1, 1, 1, 1], pairs=pairs, max_cluster=2)

    assert max(pd.Series(rep).value_counts()) <= 2


def test_strongest_pairs_merge_first():
    skills = ["a", "b", "c"]
    pairs = _pairs((1, 2, 0.95), (0, 1, 0.99))

    rep = merge_clusters(skills, counts=[3, 2, 1], pairs=pairs, max_cluster=2)

    assert rep["b"] == "a" and rep["c"] == "c"


def test_forbidden_pair_is_not_merged_even_through_a_chain():
    skills = ["java", "java core", "javascript"]
    pairs = _pairs((0, 2, 0.99), (0, 1, 0.98), (1, 2, 0.97))

    rep = merge_clusters(skills, counts=[10, 1, 10], pairs=pairs)

    assert rep["java"] != rep["javascript"]
    assert rep["java core"] == "java"


def test_different_numbers_are_not_merged():
    rep = merge_clusters(["2d", "3d"], counts=[1, 1], pairs=_pairs((0, 1, 0.99)))

    assert rep["2d"] != rep["3d"]


def test_different_languages_are_not_merged():
    skills = ["tiếng anh", "tiếng trung"]

    rep = merge_clusters(skills, counts=[1, 1], pairs=_pairs((0, 1, 0.99)))

    assert rep["tiếng anh"] != rep["tiếng trung"]


def test_embedding_tier_only_merges_same_kind_and_lists_changed_skills():
    skills = pd.DataFrame(
        {
            "skill": ["react", "reactjs", "giao tiếp", "giao tiếp tốt với khách"],
            "count": [50, 5, 100, 2],
            "kind": ["technical", "technical", "soft", "technical"],
        }
    )
    vectors = _unit([1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.99, 0.1])

    merges = embedding_tier(skills, vectors, threshold=0.9)

    assert merges[["skill", "canonical"]].values.tolist() == [["reactjs", "react"]]
    assert list(merges.columns) == ["skill", "canonical", "tier", "score"]
    assert set(merges.tier) == {"embedding"}


def test_chain_is_cut_when_a_member_drifts_too_far_from_the_representative():
    # a≈b và b≈c đều được duyệt, nhưng c cách xa a (đại diện của cụm).
    vectors = _unit([1, 0, 0], [0.8, 0.6, 0], [0.3, 0.95, 0])
    pairs = _pairs((0, 1, 0.8), (1, 2, 0.75))

    rep = merge_clusters(["a", "b", "c"], [10, 5, 1], pairs, vectors=vectors, min_rep_cosine=0.7)

    assert rep["b"] == "a"
    assert rep["c"] == "c"  # cos(a, c) ≈ 0.3 < 0.7


def test_without_floor_the_same_chain_is_merged():
    pairs = _pairs((0, 1, 0.8), (1, 2, 0.75))

    rep = merge_clusters(["a", "b", "c"], [10, 5, 1], pairs)

    assert rep["c"] == "a"
