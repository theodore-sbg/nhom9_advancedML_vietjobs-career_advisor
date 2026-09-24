"""CV Agent: LLM đọc CV, trích kỹ năng, nghề muốn làm, tỉnh và số năm kinh nghiệm.

LLM chỉ trích chữ. Code nối từng trường vào node của đồ thị; tên không có trong đồ thị thì bỏ.
Khác với nối thực thể bằng code trên cả CV, LLM phân biệt được nghề đang làm với nghề muốn làm.
"""

from __future__ import annotations

import json

import networkx as nx

from career_advisor.cleaning.fields import experience_level
from career_advisor.cleaning.locations import FOREIGN, UNKNOWN, province_of
from career_advisor.graph import query as q
from career_advisor.llm import extract_json
from career_advisor.rag.linking import LinkedEntities, link_entities
from career_advisor.retrieval.hybrid import extract_skills

SYSTEM = "Bạn đọc CV tiếng Việt và trích thông tin. Chỉ trả JSON."
PROMPT = """Đọc CV dưới đây và trích thông tin. Chỉ lấy điều CV nói rõ, không đoán.
- skills: các kỹ năng, công cụ, phần mềm người này đã có. Mỗi kỹ năng viết ngắn (ví dụ "excel", "đàm phán").
- target_title: nghề hoặc vị trí người này MUỐN làm. Nếu CV không nói muốn làm gì thì lấy nghề hiện tại.
- province: tỉnh hoặc thành phố muốn làm, không có thì null.
- years_experience: tổng số năm kinh nghiệm, không có thì null.
Trả về JSON: {"skills": [], "target_title": null, "province": null, "years_experience": null}

CV:
"""


def _read(text: str) -> dict:
    try:
        data = json.loads(extract_json(text))
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _skills(G: nx.DiGraph, names: list) -> list[str]:
    found: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            continue
        resolved = q.resolve_skill(G, name)
        found |= {resolved} if resolved else extract_skills(G, name.lower())
    return sorted(found)


def extract_profile(client, G: nx.DiGraph, cv_text: str) -> LinkedEntities:
    data = _read(client.complete(PROMPT + cv_text, system=SYSTEM, json_output=True))
    linked = LinkedEntities(skills=_skills(G, data.get("skills") or []))

    title = data.get("target_title")
    if isinstance(title, str) and title.strip():
        target = link_entities(G, f"muốn làm {title}")
        linked.titles, linked.categories = target.titles[:1], target.categories[:1]

    place = data.get("province")
    province = province_of(place) if isinstance(place, str) else None
    if province == FOREIGN:
        linked.foreign = True
    elif province and province != UNKNOWN:
        linked.provinces = [province]

    years = data.get("years_experience")
    if isinstance(years, int | float) and years >= 0:
        linked.experience_levels = [experience_level(years * 12)]
    linked.skills = [s for s in linked.skills if s not in linked.titles]
    return linked
