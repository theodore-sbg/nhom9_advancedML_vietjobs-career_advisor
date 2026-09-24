import pytest

from career_advisor.evaluation.retrieval_eval import pool, score_systems

RANKINGS = {
    "cv1": {"a": [1, 2, 3], "b": [3, 4, 5]},
    "cv2": {"a": [7, 8], "b": [9, 7]},
}


def test_pool_is_the_union_of_every_system_top_k():
    assert pool(RANKINGS) == {"cv1": [1, 2, 3, 4, 5], "cv2": [7, 8, 9]}


def test_score_systems_averages_per_cv_and_skips_undefined():
    grades = {"cv1": {1: 2, 2: 0, 3: 1, 4: 2, 5: 0}, "cv2": {7: 0, 8: 0, 9: 0}}

    table = score_systems(RANKINGS, grades, k=3).set_index("system")

    # cv2 không có tin phù hợp: recall và nDCG không xác định, MRR = 0.
    assert table.loc["a", "recall@k"] == pytest.approx(1 / 2)  # chỉ cv1: có tin 1, thiếu tin 4
    assert table.loc["b", "mrr"] == pytest.approx((1 / 2 + 0) / 2)  # cv1: tin 4 ở hạng 2
    assert table.loc["a", "n_cv"] == 2
