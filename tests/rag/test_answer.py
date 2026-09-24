import json

import pandas as pd

from career_advisor.llm import FakeLLMClient
from career_advisor.rag.answer import answer_question, read_answer
from career_advisor.rag.vector_rag import posting_context, vector_answer


def _reply(answer="Lương trung vị 12,5 triệu [#1].", refuse=False, citations=(1,)):
    return lambda prompt: json.dumps({"answer": answer, "refuse": refuse, "citations": list(citations)})


def test_answer_uses_facts_and_parses_json(small_graph):
    client = FakeLLMClient(_reply())

    ans = answer_question(client, small_graph, "Lương kế toán tổng hợp ở Bắc Ninh bao nhiêu?")

    prompt = client.prompts[0]
    assert "Lương trung vị của kế toán tổng hợp ở Bắc Ninh" in prompt and "mã tin: #" in prompt
    # Dữ kiện không được đánh số, để LLM không trích nhầm số thứ tự dữ kiện làm mã tin.
    assert "[F1]" not in prompt
    assert ans.text == "Lương trung vị 12,5 triệu [#1]." and not ans.refuse and ans.citations == [1]
    assert ans.llm_called and ans.facts


def test_no_fact_means_refusal_without_calling_the_llm(small_graph):
    client = FakeLLMClient(_reply())

    ans = answer_question(client, small_graph, "Thời tiết hôm nay thế nào?")

    assert ans.refuse and not ans.llm_called and client.calls == 0


def test_out_of_scope_is_refused_without_calling_the_llm(small_graph):
    client = FakeLLMClient(_reply())

    ans = answer_question(client, small_graph, "Lương kế toán tổng hợp năm 2023?")

    assert ans.refuse and client.calls == 0
    assert "2025" in ans.text


def test_read_answer_falls_back_on_broken_json():
    text, refuse, citations = read_answer("Lương khoảng 12 triệu [#5], [#7]")

    assert text.startswith("Lương") and not refuse and citations == [5, 7]


def test_posting_context_shows_salary_or_negotiable():
    base = {"job_title": "Kế toán", "provinces": ["hà nội"], "experience_level": "1_2y"}
    paid = pd.Series({**base, "salary_min": 10.0, "salary_max": 15.0})
    negotiable = pd.Series({**base, "salary_min": float("nan"), "salary_max": float("nan")})

    assert "10–15 triệu" in posting_context(7, paid, ["excel"]) and "[#7]" in posting_context(7, paid, [])
    assert "thoả thuận" in posting_context(7, negotiable, [])


class FixedSearcher:
    def search(self, query, k=10):
        return pd.DataFrame({"posting_id": [0, 1], "score": [0.9, 0.8]})


def test_vector_answer_gives_retrieved_postings_to_the_llm():
    postings = pd.DataFrame(
        {
            "job_title": ["Kế toán", "Kế toán thuế"],
            "provinces": [["hà nội"], ["bắc ninh"]],
            "experience_level": ["1_2y", "none"],
            "salary_min": [10.0, 12.0],
            "salary_max": [15.0, 14.0],
        }
    )
    client = FakeLLMClient(_reply(citations=(0, 1)))

    ans = vector_answer(client, FixedSearcher(), postings, {0: ["excel"]}, "Lương kế toán?", k=2)

    assert "[#0] Kế toán" in client.prompts[0] and "[#1] Kế toán thuế" in client.prompts[0]
    assert ans.citations == [0, 1] and ans.llm_called
