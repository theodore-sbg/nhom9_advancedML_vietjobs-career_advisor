import pytest

from career_advisor.evaluation.metrics import precision_recall


def test_precision_recall_matches_hand_count():
    truth = [True, True, True, False, False, True]
    pred = [True, False, True, True, False, True]

    m = precision_recall(truth, pred)

    assert (m["tp"], m["fp"], m["fn"], m["n"]) == (3, 1, 1, 6)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)


def test_no_positive_prediction_gives_none_precision():
    m = precision_recall([True, False], [False, False])

    assert m["precision"] is None and m["recall"] == 0.0


def test_ranking_metrics_match_hand_computation():
    from career_advisor.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

    ranked = [5, 3, 9, 1]
    grades = {3: 2, 1: 1, 7: 2}  # tin 7 phù hợp nhưng không được xếp; tin 5, 9 điểm 0

    # "Phù hợp" = điểm 2: có 2 tin (3 và 7), xếp được 1 tin trong top 4.
    assert recall_at_k(ranked, grades, k=4) == pytest.approx(1 / 2)
    assert mrr(ranked, grades) == pytest.approx(1 / 2)  # tin phù hợp đầu tiên ở hạng 2
    import math

    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    ideal = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, grades, k=4) == pytest.approx(dcg / ideal)


def test_ranking_metrics_with_no_relevant_posting():
    from career_advisor.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

    assert recall_at_k([1, 2], {}, k=2) is None
    assert mrr([1, 2], {1: 0}) == 0.0
    assert ndcg_at_k([1, 2], {1: 0}, k=2) is None


def test_cohen_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    from career_advisor.evaluation.metrics import cohen_kappa

    a = [0, 1, 2, 2, 1, 0, 2, 1]
    b = [0, 1, 2, 1, 1, 0, 2, 2]

    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    assert cohen_kappa(a, b, weights="linear") == pytest.approx(cohen_kappa_score(a, b, weights="linear"))


def test_paired_bootstrap_on_identical_scores_finds_no_difference():
    from career_advisor.evaluation.metrics import paired_bootstrap

    result = paired_bootstrap([0.5, 0.7, 0.9, 0.6], [0.5, 0.7, 0.9, 0.6])

    assert result["mean_diff"] == 0
    assert result["ci_low"] <= 0 <= result["ci_high"]
    assert result["p_value"] == 1.0


def test_paired_bootstrap_detects_a_consistent_gain():
    from career_advisor.evaluation.metrics import paired_bootstrap

    a = [0.80, 0.75, 0.90, 0.85, 0.70, 0.95, 0.88, 0.79]
    b = [x - 0.1 for x in a]
    result = paired_bootstrap(a, b)

    assert abs(result["mean_diff"] - 0.1) < 1e-9
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01
    assert result["n"] == 8


def test_paired_bootstrap_is_deterministic_for_a_seed():
    from career_advisor.evaluation.metrics import paired_bootstrap

    a, b = [0.3, 0.9, 0.5, 0.7, 0.2], [0.4, 0.6, 0.5, 0.9, 0.1]
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)


def test_paired_bootstrap_rejects_unpaired_input():
    import pytest

    from career_advisor.evaluation.metrics import paired_bootstrap

    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])
