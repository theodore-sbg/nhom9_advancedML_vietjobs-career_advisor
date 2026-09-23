import json
import re

import numpy as np
import pandas as pd

from career_advisor.cleaning.resolve import judge_pairs, parse_judgements, resolve_tiers
from career_advisor.llm import FakeLLMClient


def _answer_all(same_if):
    """LLM giả: đọc các dòng "k. a | b" trong prompt và trả JSON."""

    def respond(prompt: str) -> str:
        rows = re.findall(r"^(\d+)\. (.+?) \| (.+)$", prompt, flags=re.M)
        return json.dumps({"answers": [{"id": int(k), "same": same_if(a, b)} for k, a, b in rows]})

    return respond


def test_parse_judgements_reads_json_and_defaults_missing_to_false():
    text = '{"answers": [{"id": 1, "same": true}, {"id": 3, "same": false}]}'

    assert parse_judgements(text, 3) == [True, False, False]


def test_parse_judgements_survives_bad_json():
    assert parse_judgements("không phải json", 2) == [False, False]


def test_judge_pairs_batches_calls():
    client = FakeLLMClient(_answer_all(lambda a, b: a[0] == b[0]))
    pairs = [("excel", "e-xcel"), ("word", "excel"), ("misa", "mi sa")] * 3

    result = judge_pairs(client, pairs, batch_size=4)

    assert result == [True, False, True] * 3
    assert client.calls == 3  # 9 cặp / 4 cặp mỗi lượt


def _skills():
    # a≈b rất gần (tự gộp), a≈c vừa (hỏi LLM), d≈e vừa (hỏi LLM), f xa.
    skills = pd.DataFrame(
        {
            "skill": ["a", "b", "c", "d", "e", "f"],
            "count": [50, 10, 20, 8, 9, 30],
            "kind": ["technical"] * 6,
        }
    )
    base = np.eye(6, dtype=np.float32)
    v = base.copy()
    v[1] = base[0] * 0.97 + base[1] * 0.243  # cos(a, b) ≈ 0.97
    v[2] = base[0] * 0.85 + base[2] * 0.527  # cos(a, c) ≈ 0.85, cos(b, c) ≈ 0.82
    v[4] = base[3] * 0.85 + base[4] * 0.527  # cos(d, e) ≈ 0.85
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return skills, v


def test_resolve_tiers_merges_auto_pairs_and_llm_approved_pairs():
    skills, v = _skills()
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return [set(p) != {"d", "e"} for p in pairs]

    merges = resolve_tiers(skills, v, auto_threshold=0.95, llm_low=0.5, judge=judge, min_llm_count=1)
    final = dict(zip(merges.skill, merges.canonical, strict=True))
    tiers = dict(zip(merges.skill, merges.tier, strict=True))

    assert final["b"] == "a" and tiers["b"] == "embedding"
    assert final["c"] == "a" and tiers["c"] == "llm"
    assert "d" not in final and "e" not in final
    # LLM thấy tên đại diện ("a"), không thấy tên đã bị gộp ("b").
    assert all("b" not in p for p in asked)


def test_resolve_tiers_only_asks_llm_about_frequent_skills():
    skills, v = _skills()
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return [True] * len(pairs)

    resolve_tiers(skills, v, auto_threshold=0.95, llm_low=0.5, judge=judge, min_llm_count=9)

    assert all(set(p) != {"d", "e"} for p in asked)  # d chỉ có 8 tin


def test_resolve_tiers_without_judge_is_embedding_only():
    skills, v = _skills()

    merges = resolve_tiers(skills, v, auto_threshold=0.95, llm_low=0.5, judge=None)

    assert set(merges.tier) == {"embedding"}


def test_parse_judgements_accepts_code_fences_and_surrounding_text():
    text = 'Đây là kết quả:\n```json\n{"answers": [{"id": 1, "same": true}]}\n```'

    assert parse_judgements(text, 1) == [True]


def test_read_judgements_reports_whether_every_pair_was_answered():
    from career_advisor.cleaning.resolve import read_judgements

    full = '{"answers": [{"id": 1, "same": true}, {"id": 2, "same": false}]}'
    partial = '{"answers": [{"id": 1, "same": true}]}'

    assert read_judgements(full, 2) == ([True, False], True)
    assert read_judgements(partial, 2) == ([True, False], False)
    assert read_judgements("hỏng", 2) == ([False, False], False)


def test_judge_prompt_lists_pairs_with_ids():
    from career_advisor.cleaning.resolve import judge_prompt

    prompt = judge_prompt([("excel", "ms excel"), ("java", "javascript")])

    assert "1. excel | ms excel" in prompt and "2. java | javascript" in prompt


def test_resolve_tiers_asks_llm_only_about_allowed_kinds():
    skills, v = _skills()
    skills.loc[skills.skill.isin(["d", "e"]), "kind"] = "soft"
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return [True] * len(pairs)

    merges = resolve_tiers(skills, v, 0.95, 0.5, judge=judge, min_llm_count=1, llm_kinds=("technical",))

    assert all(set(p) != {"d", "e"} for p in asked)
    assert "e" not in set(merges.skill) and "d" not in set(merges.skill)
    assert ("a", "c") in asked or ("c", "a") in asked


def test_judge_pairs_splits_a_batch_that_fails_to_generate():
    from career_advisor.llm import LLMGenerationError

    good = _answer_all(lambda a, b: True)

    def respond(prompt: str) -> str:
        if "bad | pair" in prompt:
            raise LLMGenerationError("token repeat limit reached")
        return good(prompt)

    client = FakeLLMClient(respond)
    pairs = [("a", "b"), ("bad", "pair"), ("c", "d"), ("e", "f")]

    result = judge_pairs(client, pairs, batch_size=4)

    # Lô 4 cặp lỗi → chia đôi; nửa có cặp hỏng chia tiếp; cặp hỏng đứng một mình thì coi là "khác".
    assert result == [True, False, True, True]
