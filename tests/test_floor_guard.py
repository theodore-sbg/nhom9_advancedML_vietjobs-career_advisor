from career_advisor.floor_guard import Line, find_violations, parse_diff

DIFF = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2  # noqa: E501
diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -3 +3 @@
-    assert x == 2
+    pass
"""


def test_parse_diff_splits_added_and_removed_lines_by_file():
    added, removed = parse_diff(DIFF)

    assert Line("src/a.py", "x = 2  # noqa: E501") in added
    assert Line("tests/test_a.py", "    assert x == 2") in removed
    assert all(line.file != "" for line in added + removed)


def _rules(added=(), removed=()):
    return [v.rule for v in find_violations(list(added), list(removed))]


def test_new_suppression_comment_is_flagged():
    assert _rules(added=[Line("src/a.py", "x = f()  # type: ignore")]) == ["silenced-checker"]
    assert _rules(added=[Line("src/a.py", "import os  # noqa")]) == ["silenced-checker"]


def test_new_skip_in_tests_is_flagged():
    assert _rules(added=[Line("tests/test_a.py", "@pytest.mark.skip(reason='flaky')")]) == [
        "test-made-easier"
    ]


def test_unfinished_stub_is_flagged():
    assert _rules(added=[Line("src/a.py", "    raise NotImplementedError")]) == ["unfinished-work"]
    assert _rules(added=[Line("src/a.py", "    # TODO viết sau")]) == ["unfinished-work"]


def test_swallowed_exception_is_flagged():
    assert _rules(added=[Line("src/a.py", "    except Exception: pass")]) == ["unfinished-work"]


def test_assertion_removed_from_test_file_is_flagged():
    assert _rules(removed=[Line("tests/test_a.py", "    assert x == 2")]) == ["assertion-removed"]


def test_assertion_removed_from_source_file_is_not_flagged():
    assert _rules(removed=[Line("src/a.py", "    assert x == 2")]) == []


def test_secret_is_flagged_without_leaking_its_value():
    key = "AIza" + "B" * 35
    [violation] = find_violations([Line("src/config.py", f'API_KEY = "{key}"')], [])

    assert violation.rule == "secret"
    assert key not in violation.text


def test_env_file_in_diff_is_flagged():
    assert _rules(added=[Line(".env", "GEMINI_API_KEY=abc")]) == ["secret"]


def test_new_exception_row_in_constraints_is_flagged():
    row = "| E2 | no-skip | tests/x.py | lý do | @theodore | 2026-12-01 |"
    assert _rules(added=[Line("CONSTRAINTS.md", row)]) == ["new-exception"]


def test_lowered_number_in_constraints_is_flagged():
    removed = [Line("CONSTRAINTS.md", "| Số test | 414 | không được giảm |")]
    added = [Line("CONSTRAINTS.md", "| Số test | 400 | không được giảm |")]
    assert _rules(added, removed) == ["threshold-lowered"]


def test_raised_number_in_constraints_is_silent():
    removed = [Line("CONSTRAINTS.md", "| Số test | 414 | không được giảm |")]
    added = [Line("CONSTRAINTS.md", "| Số test | 420 | không được giảm |")]
    assert _rules(added, removed) == []


def test_guard_does_not_flag_its_own_pattern_definitions():
    assert (
        _rules(added=[Line("src/career_advisor/floor_guard.py", 'SKIPS = re.compile(r"@pytest.mark.skip")')])
        == []
    )


def test_clean_change_has_no_violations():
    assert _rules(added=[Line("src/a.py", "def f():"), Line("src/a.py", "    return 1")]) == []


def test_code_patterns_mentioned_in_docs_are_not_flagged():
    doc = "- Không thêm `# noqa`, `@pytest.mark.skip`, `raise NotImplementedError`, `TODO`."
    assert _rules(added=[Line("CONSTRAINTS.md", doc), Line("docs/notes.md", doc)]) == []


def test_secret_in_docs_is_still_flagged():
    assert _rules(added=[Line("README.md", "sk-ant-" + "x" * 30)]) == ["secret"]


def test_editing_an_assertion_is_not_flagged():
    removed = [Line("tests/test_a.py", "    assert x == 2")]
    added = [Line("tests/test_a.py", "    assert x == 3")]
    assert _rules(added, removed) == []


def test_net_loss_of_assertions_in_a_test_file_is_flagged():
    removed = [Line("tests/test_a.py", "    assert x == 2"), Line("tests/test_a.py", "    assert y == 1")]
    added = [Line("tests/test_a.py", "    assert x == 3")]
    assert _rules(added, removed) == ["assertion-removed"]


def test_lowering_a_must_not_rise_metric_is_silent():
    removed = [Line("CONSTRAINTS.md", "| Chỗ skipif | 3 | không được tăng |")]
    added = [Line("CONSTRAINTS.md", "| Chỗ skipif | 2 | không được tăng |")]
    assert _rules(added, removed) == []


def test_raising_a_must_not_rise_metric_is_flagged():
    removed = [Line("CONSTRAINTS.md", "| Chỗ skipif | 3 | không được tăng |")]
    added = [Line("CONSTRAINTS.md", "| Chỗ skipif | 5 | không được tăng |")]
    assert _rules(added, removed) == ["threshold-lowered"]


def test_changing_a_row_without_direction_is_silent():
    removed = [Line("CONSTRAINTS.md", "| Chỉ số | Hôm nay (2026-09-26) | Hướng |")]
    added = [Line("CONSTRAINTS.md", "| Chỉ số | Hôm nay (2026-01-01) | Hướng |")]
    assert _rules(added, removed) == []


def test_removing_an_assertion_from_the_guards_own_tests_is_flagged():
    removed = [Line("tests/test_floor_guard.py", '    assert violation.rule == "secret"')]
    assert _rules(removed=removed) == ["assertion-removed"]


def test_parse_diff_keeps_content_lines_that_look_like_file_headers():
    diff = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n"
        "--- a/tests/test_a.py\n"
        "+++ b/tests/test_a.py\n"
        "@@ -1,2 +1,1 @@\n"
        "--- ghi chú\n"
        "+++ tiêu đề\n"
        "-    assert x\n"
    )
    added, removed = parse_diff(diff)

    assert removed == [Line("tests/test_a.py", "-- ghi chú"), Line("tests/test_a.py", "    assert x")]
    assert added == [Line("tests/test_a.py", "++ tiêu đề")]
