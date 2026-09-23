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
