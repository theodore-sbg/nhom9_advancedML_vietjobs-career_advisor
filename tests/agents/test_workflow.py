import json
import re

from career_advisor.agents.cv_agent import extract_profile
from career_advisor.agents.workflow import MultiAgent, SingleAgent
from career_advisor.llm import FakeLLMClient

QUESTION = "Lương trung vị của kế toán tổng hợp ở Bắc Ninh là bao nhiêu?"
CV = "Kế toán viên 2 năm ở Bắc Ninh, dùng MS Excel thành thạo, cẩn thận. Muốn làm kế toán tổng hợp."
PROFILE = {
    "skills": ["MS Excel", "cẩn thận", "bơi lội"],
    "target_title": "Kế toán tổng hợp",
    "province": "Bắc Ninh",
    "years_experience": 2,
}
GOOD = "Lương trung vị là 12,5 triệu đồng/tháng"
BAD = "Lương trung vị là 99 triệu đồng/tháng"


def _first_source(prompt: str) -> list[int]:
    match = re.search(r"mã tin: #(\d+)", prompt)
    return [int(match[1])] if match else []


def _llm(answers: list[str], profile: dict | None = None) -> FakeLLMClient:
    """CV Agent nhận `profile`. Planner nhận lần lượt các câu trong `answers` (câu cuối lặp lại)."""
    planner_calls = []

    def respond(prompt: str) -> str:
        if "CV:" in prompt and "Dữ kiện:" not in prompt:
            return json.dumps(profile or PROFILE, ensure_ascii=False)
        text = answers[min(len(planner_calls), len(answers) - 1)]
        planner_calls.append(prompt)
        return json.dumps({"answer": text, "refuse": False, "citations": _first_source(prompt)})

    return FakeLLMClient(respond)


def test_cv_agent_maps_llm_fields_to_graph_nodes(small_graph):
    linked = extract_profile(_llm([GOOD]), small_graph, CV)

    # "MS Excel" nối qua alias về "excel"; "bơi lội" không có trong đồ thị nên bị bỏ.
    assert linked.skills == ["cẩn thận", "excel"]
    assert linked.titles == ["kế toán tổng hợp"]
    assert linked.provinces == ["bắc ninh"]
    assert linked.experience_levels == ["1_2y"]


def test_cv_agent_survives_broken_json(small_graph):
    client = FakeLLMClient(lambda prompt: "không phải JSON")

    linked = extract_profile(client, small_graph, CV)

    assert linked.skills == [] and linked.titles == []


def test_question_skips_cv_agent_and_passes_verifier(small_graph):
    client = _llm([GOOD])

    result = MultiAgent(client, small_graph).run(question=QUESTION)

    assert result.trace == ["graph", "planner", "verifier"]
    assert result.verification.ok and result.attempts == 1 and not result.fallback
    assert result.answer.text.startswith(GOOD) and result.llm_calls == 1


def test_rejected_answer_goes_back_to_planner_with_the_reasons(small_graph):
    client = _llm([BAD, GOOD])

    result = MultiAgent(client, small_graph).run(question=QUESTION)

    assert result.trace == ["graph", "planner", "verifier", "planner", "verifier"]
    assert "99" in client.prompts[1] and "Verifier" in client.prompts[1]
    assert result.verification.ok and result.attempts == 2 and result.answer.text.startswith(GOOD)


def test_second_rejection_falls_back_to_the_raw_facts(small_graph):
    client = _llm([BAD])

    result = MultiAgent(client, small_graph).run(question=QUESTION)

    assert result.attempts == 2 and result.fallback and result.llm_calls == 2
    assert "99" not in result.answer.text and "12,5" in result.answer.text
    assert result.verification.ok


def test_cv_runs_all_four_agents_and_plans_from_missing_skills(small_graph):
    client = _llm([GOOD])

    result = MultiAgent(client, small_graph).run(cv_text=CV)

    assert result.trace == ["cv", "graph", "planner", "verifier"]
    assert result.answer.linked.titles == ["kế toán tổng hợp"]
    kinds = {f.kind for f in result.answer.facts}
    assert {"salary", "missing_skills"} <= kinds
    assert "lộ trình" in client.prompts[-1] and CV in client.prompts[-1]
    assert result.llm_calls == 2


def test_numbers_from_the_cv_itself_are_not_rejected(small_graph):
    client = _llm(["Bạn từng xử lý 347 hồ sơ mỗi tháng. " + GOOD])

    result = MultiAgent(client, small_graph).run(cv_text=CV + " Từng xử lý 347 hồ sơ mỗi tháng.")

    assert result.verification.ok and result.attempts == 1


def test_out_of_scope_question_is_refused_without_llm(small_graph):
    client = _llm([GOOD])

    result = MultiAgent(client, small_graph).run(question="Lương kế toán tổng hợp năm 2023?")

    assert result.answer.refuse and result.llm_calls == 0 and client.calls == 0


def test_single_agent_makes_one_call_and_keeps_unverified_numbers(small_graph):
    client = _llm([BAD])

    result = SingleAgent(client, small_graph).run(cv_text=CV)

    assert result.trace == ["single"] and result.llm_calls == 1
    assert result.answer.text.startswith(BAD) and not result.verification.ok


def test_single_and_multi_agent_send_the_same_first_prompt_for_questions(small_graph):
    single, multi = _llm([GOOD]), _llm([GOOD])

    SingleAgent(single, small_graph).run(question=QUESTION)
    MultiAgent(multi, small_graph).run(question=QUESTION)

    assert single.prompts == multi.prompts
