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


def test_labeled_precision_scores_each_system_on_the_labeled_pairs_in_its_top_k():
    from career_advisor.evaluation.retrieval_eval import labeled_precision

    rankings = {"cv1": {"a": [1, 2, 3], "b": [9, 8, 1]}, "cv2": {"a": [5], "b": [6]}}
    labels = {"cv1": {1: 2, 2: 0, 8: 1}, "cv2": {6: 2}}

    table = labeled_precision(rankings, labels, k=3).set_index("system")

    # a: nhãn của #1 (2), #2 (0) → 2 cặp; b: #8 (1), #1 (2), #6 (2) → 3 cặp.
    assert table.loc["a", "n_labeled"] == 2 and table.loc["b", "n_labeled"] == 3
    assert table.loc["a", "share_relevant"] == 0.5 and table.loc["a", "share_partly"] == 0.5
    assert table.loc["b", "share_relevant"] == 2 / 3 and table.loc["b", "share_partly"] == 1.0
    assert table.loc["b", "mean_grade"] == 5 / 3
