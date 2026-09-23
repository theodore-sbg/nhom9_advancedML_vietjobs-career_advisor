import numpy as np
import pandas as pd

from career_advisor.evaluation.entity_resolution import choose_threshold, labeled_pairs, predict_same


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3", "p4", "p5"],
            "a": ["ReactJS", "excel", "java", "misa", "bán hàng"],
            "b": ["React.js", "ms excel", "javascript", "phần mềm kế toán misa", "tư vấn"],
            "stratum": ["rule", "cosine_4", "cosine_3", "cosine_1", "cosine_0"],
            "cosine": [np.nan, 0.97, 0.93, 0.86, 0.80],
            "split": ["dev", "dev", "test", "test", "dev"],
            "label": ["same", "same", "different", "same", "skip"],
        }
    )


def test_labeled_pairs_drop_skips_and_add_truth():
    pairs = labeled_pairs(_labels())

    assert "p5" not in set(pairs.pair_id)
    assert pairs.set_index("pair_id").truth.to_dict() == {"p1": True, "p2": True, "p3": False, "p4": True}


def test_predict_same_follows_rules_then_merges():
    pairs = labeled_pairs(_labels())
    merges = pd.DataFrame({"skill": ["phần mềm kế toán misa"], "canonical": ["misa"]})

    rule_only = predict_same(pairs, merges.iloc[0:0])
    with_merges = predict_same(pairs, merges)

    # "ReactJS" và "React.js" gộp nhờ luật; "excel" và "ms excel" cũng vậy (bóc tên hãng).
    assert rule_only.tolist() == [True, True, False, False]
    assert with_merges.tolist() == [True, True, False, True]


def test_choose_threshold_picks_lowest_value_reaching_precision():
    truth = pd.Series([True, True, True, False, True, False])
    score = pd.Series([0.99, 0.95, 0.91, 0.90, 0.86, 0.80])

    best, table = choose_threshold(truth, lambda t: score >= t, [0.80, 0.85, 0.90, 0.95], min_precision=0.95)

    # 0.85 và 0.90 đều gồm cặp sai 0.90; 0.95 là giá trị thấp nhất đạt precision 1.0.
    assert best == 0.95
    assert list(table.columns[:3]) == ["threshold", "precision", "recall"]


def test_choose_threshold_returns_none_when_nothing_is_precise_enough():
    truth = pd.Series([False, False])

    best, _ = choose_threshold(truth, lambda t: pd.Series([True, True]), [0.5], min_precision=0.95)

    assert best is None
