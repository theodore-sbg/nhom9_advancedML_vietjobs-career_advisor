"""Vector RAG làm mốc so sánh: lấy k tin gần câu hỏi nhất bằng bge-m3, LLM tự đọc và tự tính."""

from __future__ import annotations

import math

import pandas as pd

from career_advisor.rag.answer import SYSTEM, Answer, read_answer

K = 8
MAX_SKILLS = 8
RULES = """Trả lời câu hỏi CHỈ dựa trên các tin tuyển dụng dưới đây, lấy từ TopCV tháng 7–10/2025.
- Mọi con số phải lấy hoặc tính được từ các tin này.
- Sau mỗi ý có số liệu, ghi mã tin làm dẫn chứng dạng [#123]. Chỉ dùng mã tin có trong danh sách.
- Nếu các tin không đủ để trả lời điều được hỏi, đặt "refuse": true và nói ngắn gọn là dữ liệu không đủ.
- Mức lương chỉ để tham khảo, không phải cam kết. Không nhắc tới giới tính hay tuổi.
Trả về JSON: {"answer": "...", "refuse": false, "citations": [123, 456]}"""


def _number(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def posting_context(pid: int, row: pd.Series, skills: list[str]) -> str:
    low, high = row["salary_min"], row["salary_max"]
    salary = "thoả thuận" if low is None or math.isnan(low) else f"{_number(low)}–{_number(high)} triệu"
    provinces = ", ".join(p.title() for p in row["provinces"])
    return (
        f"[#{pid}] {row['job_title']} | nơi làm: {provinces} | lương: {salary} | "
        f"kinh nghiệm: {row['experience_level']} | kỹ năng: {', '.join(skills[:MAX_SKILLS])}"
    )


def vector_answer(client, searcher, postings: pd.DataFrame, skills_of, question: str, k: int = K) -> Answer:
    ids = searcher.search(question, k=k)["posting_id"].astype(int).tolist()
    context = "\n".join(posting_context(pid, postings.loc[pid], list(skills_of.get(pid, []))) for pid in ids)
    raw = client.complete(
        f"{RULES}\n\nTin tuyển dụng:\n{context}\n\nCâu hỏi: {question}", system=SYSTEM, json_output=True
    )
    text, refuse, citations = read_answer(raw)
    return Answer(text, refuse, citations, llm_called=True, allowed_ids=ids)
