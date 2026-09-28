import json

from career_advisor.config import EVAL_DIR
from career_advisor.evaluation.summary import SECTIONS, build_report, md_table

COLS = [("Cách", "system", "text"), ("Recall@10", "recall@k", "pct"), ("Số CV", "n_cv", "int")]


def test_md_table_formats_percent_with_comma_and_ints_plainly():
    table = md_table([{"system": "bm25", "recall@k": 0.4576, "n_cv": 20}], COLS)

    assert table.splitlines() == [
        "| Cách | Recall@10 | Số CV |",
        "|---|---|---|",
        "| bm25 | 45,8% | 20 |",
    ]


def test_md_table_shows_dash_for_missing_or_nan_values():
    rows = [{"system": "x", "recall@k": None}, {"system": "y", "recall@k": float("nan"), "n_cv": 5}]
    lines = md_table(rows, COLS).splitlines()

    assert lines[2] == "| x | — | — |"
    assert lines[3] == "| y | — | 5 |"


def test_md_table_formats_plain_numbers_and_seconds():
    cols = [("F1", "f1", "num"), ("Thời gian", "seconds", "sec")]
    assert md_table([{"f1": 0.8167, "seconds": 1898.0}], cols).splitlines()[2] == "| 0,817 | 1898 s |"


def _retrieval():
    return {
        "commit": "76356bf",
        "rows": [
            {
                "system": "dense",
                "recall@k": 0.535,
                "mrr": 0.875,
                "ndcg@k": 0.807,
                "n_cv": 20,
                "split": "test",
            },
            {"system": "dense", "recall@k": 0.427, "mrr": 0.772, "ndcg@k": 0.737, "n_cv": 20, "split": "dev"},
        ],
        "human_check": {"test": [{"system": "dense", "n_labeled_human": 24, "share_relevant_human": 0.833}]},
        "agreement": {
            "human_vs_llm": {
                "n": 99,
                "unweighted": 0.004,
                "exact_agreement": 0.424,
                "binary": {">=2": {"kappa": -0.068, "agreement": 0.465}},
            }
        },
    }


def test_report_contains_retrieval_table_for_test_split_only():
    report = build_report({"retrieval": _retrieval()})

    assert "## Truy xuất CV → tin" in report
    assert "53,5%" in report
    assert "42,7%" not in report


def test_report_cites_the_commit_that_produced_each_table():
    assert "76356bf" in build_report({"retrieval": _retrieval()})


def test_report_marks_missing_result_files_instead_of_failing():
    report = build_report({})

    assert "## Truy xuất CV → tin" in report
    assert "chưa có `eval/results/retrieval.json`" in report


def test_report_builds_every_section_from_the_real_result_files():
    results = {
        name: json.loads((EVAL_DIR / "results" / f"{name}.json").read_text().replace("NaN", "null"))
        for _, name, _ in SECTIONS
    }
    report = build_report(results)

    assert "chưa có" not in report
    assert report.count("\n## ") == len(SECTIONS)


def test_thresholds_are_shown_without_rounding():
    er = {"auto_threshold": 0.9, "llm_low": 0.78, "rows": []}
    assert "hỏi LLM từ 0,78" in build_report({"entity_resolution": er})


def test_averages_are_shown_with_one_decimal():
    cols = [("Lượt LLM", "calls", "avg"), ("RAM", "ram", "avg")]
    assert md_table([{"calls": 1.0, "ram": 5.64}], cols).splitlines()[2] == "| 1 | 5,6 |"
