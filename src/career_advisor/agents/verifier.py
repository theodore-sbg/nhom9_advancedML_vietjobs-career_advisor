"""Verifier Agent: kiểm câu trả lời bằng code trước khi đưa cho người dùng.

- Mọi con số phải có trong dữ kiện (hoặc trong chính câu hỏi, như "3 năm kinh nghiệm").
  Sai số chỉ đủ để bỏ qua cách viết (12,5 và 12.500.000 đồng); làm tròn khác đi cũng bị chặn.
- Mọi mã tin được trích phải nằm trong nguồn của dữ kiện.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from career_advisor.rag.numbers import extract_numbers
from career_advisor.rag.subgraph import Fact

ABS_TOLERANCE = 0.05
REL_TOLERANCE = 0.005

_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", flags=re.M)
_INLINE_CITATION = re.compile(r"#\s*(\d+)")


@dataclass
class Verification:
    ok: bool
    unsupported_numbers: list[float] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _supported(x: float, allowed: list[float]) -> bool:
    return any(abs(x - a) <= max(ABS_TOLERANCE, REL_TOLERANCE * abs(a)) for a in allowed)


def verify(text: str, citations: list[int], facts: list[Fact], question: str = "") -> Verification:
    allowed = [v for f in facts for v in f.values]
    allowed += [v for f in facts for v in extract_numbers(f.text)]
    allowed += extract_numbers(question)
    numbers = extract_numbers(_LIST_MARKER.sub("", text))
    bad_numbers = [x for x in numbers if not _supported(x, allowed)]

    sources = {pid for f in facts for pid in f.sources}
    cited = list(dict.fromkeys([*citations, *(int(c) for c in _INLINE_CITATION.findall(text))]))
    bad_citations = [pid for pid in cited if pid not in sources]

    reasons = [f"Con số {x:g} không có trong dữ kiện." for x in bad_numbers]
    reasons += [f"Mã tin #{pid} không có trong nguồn của dữ kiện." for pid in bad_citations]
    return Verification(not reasons, bad_numbers, bad_citations, reasons)
