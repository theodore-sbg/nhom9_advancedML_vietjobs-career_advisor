"""Planner Agent: viết câu trả lời hoặc lộ trình học, chỉ từ dữ kiện của Graph Agent.

Với câu hỏi, Planner dùng đúng prompt hỏi đáp của KG-RAG (Task 16). Với CV, Planner viết lộ trình:
kỹ năng còn thiếu có tỷ lệ tin yêu cầu cao học trước, rồi tới kỹ năng hay đi cùng.
Khi Verifier chặn, Planner viết lại một lần, kèm lý do bị chặn.
"""

from __future__ import annotations

import networkx as nx

from career_advisor.agents.verifier import Verification
from career_advisor.graph import query as q
from career_advisor.rag.answer import build_prompt, facts_block
from career_advisor.rag.linking import LinkedEntities
from career_advisor.rag.subgraph import Fact, group_node, related_fact, retrieve_facts

MAX_PLAN_RELATED = 3

PLAN_RULES = """Viết lộ trình học cho người có CV dưới đây. CHỈ dựa trên các dữ kiện lấy từ tin tuyển dụng
TopCV tháng 7–10/2025.
- Câu đầu tiên: mức lương tham khảo (lương trung vị, khoảng giữa, số tin), nếu dữ kiện có. Mức lương chỉ để
  tham khảo, không phải cam kết.
- Sau đó nêu 3–5 kỹ năng còn thiếu nên học, kỹ năng có tỷ lệ tin yêu cầu cao học trước. Ưu tiên kỹ năng chuyên
  môn; các kỹ năng mềm gom lại thành một ý. Mỗi kỹ năng ghi tỷ lệ tin yêu cầu, và nếu dữ kiện có thì nêu thêm
  kỹ năng hay đi cùng để học kế tiếp.
- Chép nguyên con số từ dữ kiện. Không tự tính, không làm tròn khác đi, không thêm con số nào khác.
- Dùng gạch đầu dòng, không viết "Bước 1", "Tháng 2" hay số thứ tự.
- Sau mỗi ý có số liệu, ghi mã tin dạng [#123]. Chỉ dùng các mã tin trong phần "(mã tin: ...)" của dữ kiện.
- Nếu dữ kiện không đủ để lập lộ trình, đặt "refuse": true và nói ngắn gọn là dữ liệu không đủ.
- Không nhắc tới giới tính hay tuổi.
Trả về JSON: {"answer": "...", "refuse": false, "citations": [123, 456]}"""

FEEDBACK = """

Câu trả lời trước của bạn bị Verifier chặn:
{previous}

Lý do:
{reasons}
Viết lại câu trả lời: bỏ hoặc sửa các con số và mã tin trên, chỉ dùng số và mã tin có trong dữ kiện."""


def graph_facts(G: nx.DiGraph, linked: LinkedEntities, plan: bool) -> list[Fact]:
    """Graph Agent. Khi lập lộ trình, thêm kỹ năng hay đi cùng của các kỹ năng còn thiếu hàng đầu."""
    facts = retrieve_facts(G, linked)
    group = group_node(linked)
    if plan and group and not any(f.kind == "out_of_scope" for f in facts):
        for stat in q.missing_skills(G, linked.skills, group, k=MAX_PLAN_RELATED):
            fact = related_fact(G, stat.skill)
            facts += [fact] if fact else []
    return facts


def plan_prompt(cv_text: str, question: str, facts: list[Fact]) -> str:
    ask = f"\n\nCâu hỏi thêm: {question}" if question else ""
    return f"{PLAN_RULES}\n\nDữ kiện:\n{facts_block(facts)}\n\nCV:\n{cv_text}{ask}"


def planner_prompt(question: str, cv_text: str, facts: list[Fact]) -> str:
    return plan_prompt(cv_text, question, facts) if cv_text else build_prompt(question, facts)


def with_feedback(prompt: str, previous: str, verification: Verification) -> str:
    reasons = "\n".join(f"- {r}" for r in verification.reasons)
    return prompt + FEEDBACK.format(previous=previous, reasons=reasons)
