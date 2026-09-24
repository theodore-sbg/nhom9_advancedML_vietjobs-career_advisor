"""Luồng 4 agent bằng LangGraph: CV → Graph → Planner → Verifier.

- CV Agent (LLM): trích kỹ năng, nghề muốn làm, tỉnh, số năm kinh nghiệm. Bỏ qua khi chỉ có câu hỏi.
- Graph Agent (code): nối thực thể vào đồ thị, lấy dữ kiện có số liệu và mã tin.
- Planner Agent (LLM): viết câu trả lời hoặc lộ trình học, chỉ từ dữ kiện.
- Verifier Agent (code): kiểm con số và mã tin. Bị chặn thì quay lại Planner kèm lý do, tối đa 1 lần.
  Vẫn bị chặn thì trả nguyên dữ kiện thay cho câu LLM viết, để không con số sai nào tới người dùng.

Bản 1 agent để so sánh: nối thực thể bằng code trên văn bản gốc (như KG-RAG ở Task 16), 1 lượt LLM,
không có Verifier chặn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TypedDict

import networkx as nx
from langgraph.graph import END, START, StateGraph

from career_advisor.agents.cv_agent import extract_profile
from career_advisor.agents.planner import graph_facts, planner_prompt, with_feedback
from career_advisor.agents.verifier import Verification, verify
from career_advisor.rag.answer import Answer, answer_from_facts, facts_block
from career_advisor.rag.linking import LinkedEntities, link_entities
from career_advisor.rag.subgraph import Fact

MAX_ATTEMPTS = 2  # lần đầu và 1 lần viết lại
FALLBACK_INTRO = "Câu trả lời tự viết chưa qua được bước kiểm tra số liệu. Đây là số liệu gốc từ dữ liệu:\n"


class AgentState(TypedDict, total=False):
    question: str
    cv_text: str
    linked: LinkedEntities
    facts: list[Fact]
    answer: Answer
    verification: Verification
    attempts: int
    llm_calls: int
    fallback: bool
    trace: list[str]


@dataclass
class AgentResult:
    answer: Answer
    verification: Verification
    attempts: int = 0
    llm_calls: int = 0
    fallback: bool = False
    trace: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _context(state: AgentState) -> str:
    """Số trong câu hỏi và CV được phép nhắc lại ("2 năm kinh nghiệm")."""
    return f"{state.get('question', '')}\n{state.get('cv_text', '')}"


def _check(answer: Answer, facts: list[Fact], context: str) -> Verification:
    return verify(answer.text, answer.citations, facts, context)


def fallback_answer(answer: Answer) -> Answer:
    usable = [f for f in answer.facts if f.kind != "out_of_scope"]
    sources = sorted({pid for f in usable for pid in f.sources})
    return Answer(
        FALLBACK_INTRO + facts_block(usable),
        False,
        sources,
        answer.facts,
        answer.linked,
        llm_called=answer.llm_called,
        allowed_ids=answer.allowed_ids,
    )


class MultiAgent:
    def __init__(self, client, G: nx.DiGraph) -> None:
        self.client, self.G = client, G
        self.app = self._build()

    def _build(self):
        client, G = self.client, self.G

        def cv_agent(state: AgentState) -> AgentState:
            linked = extract_profile(client, G, state["cv_text"])
            return {"linked": linked, "llm_calls": state["llm_calls"] + 1, "trace": [*state["trace"], "cv"]}

        def graph_agent(state: AgentState) -> AgentState:
            linked = state.get("linked") or link_entities(G, state["question"])
            facts = graph_facts(G, linked, plan=bool(state["cv_text"]))
            return {"linked": linked, "facts": facts, "trace": [*state["trace"], "graph"]}

        def planner(state: AgentState) -> AgentState:
            prompt = planner_prompt(state["question"], state["cv_text"], state["facts"])
            if state["attempts"]:
                prompt = with_feedback(prompt, state["answer"].text, state["verification"])
            answer = answer_from_facts(client, state["question"], state["facts"], state["linked"], prompt)
            return {
                "answer": answer,
                "attempts": state["attempts"] + 1,
                "llm_calls": state["llm_calls"] + int(answer.llm_called),
                "trace": [*state["trace"], "planner"],
            }

        def verifier(state: AgentState) -> AgentState:
            answer = state["answer"]
            verification = _check(answer, state["facts"], _context(state))
            update: AgentState = {"verification": verification, "trace": [*state["trace"], "verifier"]}
            if not verification.ok and state["attempts"] >= MAX_ATTEMPTS:
                answer = fallback_answer(answer)
                update |= {
                    "answer": answer,
                    "fallback": True,
                    "verification": _check(answer, state["facts"], _context(state)),
                }
            return update

        def after_verifier(state: AgentState) -> str:
            done = state["verification"].ok or state["attempts"] >= MAX_ATTEMPTS
            return END if done else "planner"

        graph = StateGraph(AgentState)
        graph.add_node("cv", cv_agent)
        graph.add_node("graph", graph_agent)
        graph.add_node("planner", planner)
        graph.add_node("verifier", verifier)
        graph.add_conditional_edges(START, lambda s: "cv" if s["cv_text"] else "graph", ["cv", "graph"])
        graph.add_edge("cv", "graph")
        graph.add_edge("graph", "planner")
        graph.add_edge("planner", "verifier")
        graph.add_conditional_edges("verifier", after_verifier, ["planner", END])
        return graph.compile()

    def run(self, question: str = "", cv_text: str = "") -> AgentResult:
        start = time.perf_counter()
        state = self.app.invoke(
            {"question": question, "cv_text": cv_text, "attempts": 0, "llm_calls": 0, "trace": []}
        )
        return AgentResult(
            state["answer"],
            state["verification"],
            state["attempts"],
            state["llm_calls"],
            state.get("fallback", False),
            state["trace"],
            time.perf_counter() - start,
        )


class SingleAgent:
    def __init__(self, client, G: nx.DiGraph) -> None:
        self.client, self.G = client, G

    def run(self, question: str = "", cv_text: str = "") -> AgentResult:
        start = time.perf_counter()
        linked = link_entities(self.G, f"{cv_text}\n{question}".strip())
        facts = graph_facts(self.G, linked, plan=bool(cv_text))
        prompt = planner_prompt(question, cv_text, facts)
        answer = answer_from_facts(self.client, question, facts, linked, prompt)
        state: AgentState = {"question": question, "cv_text": cv_text}
        return AgentResult(
            answer,
            _check(answer, facts, _context(state)),
            attempts=1,
            llm_calls=int(answer.llm_called),
            trace=["single"],
            seconds=time.perf_counter() - start,
        )
