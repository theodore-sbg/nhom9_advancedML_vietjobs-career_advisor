"""KG-RAG: nối thực thể → lấy dữ kiện từ đồ thị → LLM diễn đạt lại, kèm mã tin.

LLM chỉ được dùng các con số có trong dữ kiện. Không có dữ kiện, hoặc câu hỏi nằm ngoài phạm vi dữ
liệu, thì code tự từ chối mà không gọi LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import networkx as nx

from career_advisor.llm import extract_json
from career_advisor.rag.linking import LinkedEntities, link_entities
from career_advisor.rag.subgraph import Fact, retrieve_facts

SYSTEM = (
    "Bạn là trợ lý tư vấn nghề nghiệp ở Việt Nam. Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng. Chỉ trả JSON."
)
RULES = """Trả lời câu hỏi CHỈ dựa trên các dữ kiện dưới đây, lấy từ tin tuyển dụng TopCV tháng 7–10/2025.
- Chép nguyên con số từ dữ kiện. Không tự tính, không làm tròn khác đi.
- Không thêm con số nào không có trong dữ kiện.
- Sau mỗi ý có số liệu, ghi mã tin làm dẫn chứng dạng [#123]. Mã tin là con số đứng sau dấu # trong phần
  "(mã tin: ...)" của dữ kiện. Chỉ dùng các mã tin đó.
- Nếu dữ kiện không đủ để trả lời điều được hỏi, đặt "refuse": true và nói ngắn gọn là dữ liệu không đủ.
- Mức lương chỉ để tham khảo, không phải cam kết. Không nhắc tới giới tính hay tuổi.
Trả về JSON: {"answer": "...", "refuse": false, "citations": [123, 456]}"""
NO_DATA = "Xin lỗi, dữ liệu tin tuyển dụng hiện có không đủ để trả lời câu hỏi này."

_CITED = re.compile(r"#\s*(\d+)")


@dataclass
class Answer:
    text: str
    refuse: bool
    citations: list[int] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    linked: LinkedEntities | None = None
    llm_called: bool = False
    allowed_ids: list[int] = field(default_factory=list)  # mã tin LLM đã được xem


def facts_block(facts: list[Fact]) -> str:
    # Không đánh số dữ kiện: từng có lần LLM trích số thứ tự dữ kiện ("F4" → 4) làm mã tin.
    lines = []
    for fact in facts:
        sources = ", ".join(f"#{pid}" for pid in fact.sources)
        lines.append(f"- {fact.text}" + (f" (mã tin: {sources})" if sources else ""))
    return "\n".join(lines)


def build_prompt(question: str, facts: list[Fact]) -> str:
    return f"{RULES}\n\nDữ kiện:\n{facts_block(facts)}\n\nCâu hỏi: {question}"


def read_answer(text: str) -> tuple[str, bool, list[int]]:
    """Đọc JSON của LLM. JSON hỏng thì coi cả chuỗi là câu trả lời và lấy mã tin bằng biểu thức chính quy."""
    try:
        data = json.loads(extract_json(text))
        answer = str(data.get("answer", "")).strip()
        citations = [int(c) for c in data.get("citations", []) if str(c).lstrip("#").isdigit()]
        return answer, bool(data.get("refuse", False)), citations
    except (ValueError, AttributeError, TypeError):
        return text.strip(), False, [int(c) for c in _CITED.findall(text)]


def answer_from_facts(
    client, question: str, facts: list[Fact], linked: LinkedEntities | None = None, prompt: str | None = None
) -> Answer:
    """`prompt` thay cho prompt hỏi đáp mặc định (Planner dùng để viết lộ trình hoặc viết lại)."""
    usable = [f for f in facts if f.kind != "out_of_scope"]
    if not usable:
        text = next((f.text for f in facts if f.kind == "out_of_scope"), NO_DATA)
        return Answer(text, True, [], facts, linked, llm_called=False)
    prompt = prompt or build_prompt(question, facts)
    raw = client.complete(prompt, system=SYSTEM, json_output=True)
    text, refuse, citations = read_answer(raw)
    allowed = sorted({pid for f in facts for pid in f.sources})
    return Answer(text, refuse, citations, facts, linked, llm_called=True, allowed_ids=allowed)


def answer_question(client, G: nx.DiGraph, question: str) -> Answer:
    linked = link_entities(G, question)
    return answer_from_facts(client, question, retrieve_facts(G, linked), linked)
