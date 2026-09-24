"""Đọc con số trong câu trả lời tiếng Việt, dùng cho Verifier và cho chấm điểm.

Quy ước tiếng Việt: dấu chấm phân cách hàng nghìn ("1.178"), dấu phẩy là thập phân ("14,5").
LLM đôi khi viết kiểu Anh ("12.5"), nên "số.1–2 chữ số" được hiểu là thập phân.
Số tiền ghi bằng đồng ("12.500.000 đồng") được đổi ra triệu, vì mọi dữ kiện tính theo triệu đồng.
Mã tin ("[#1234]") và năm ("năm 2025") không phải số liệu.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[#?\s*\d+(?:\s*,\s*#?\s*\d+)*\]|#\d+")
_YEAR = re.compile(r"(?:năm|/)\s*(?:19|20)\d{2}\b")
_TR = re.compile(r"(\d+)\s*tr\s*(\d+)\b")  # "12tr5" = 12,5 triệu
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_DONG = re.compile(r"\s*(?:đồng|vnđ|vnd|đ)\b", flags=re.I)


def _parse(token: str) -> float:
    if _THOUSANDS.fullmatch(token):
        return float(token.replace(".", "").replace(",", "."))
    if "," in token:
        return float(token.replace(".", "").replace(",", "."))
    return float(token)


def extract_numbers(text: str) -> list[float]:
    text = _CITATION.sub(" ", text)
    text = _YEAR.sub(" ", text)
    text = _TR.sub(lambda m: f"{m[1]},{m[2]} triệu", text)
    values = []
    for match in _NUMBER.finditer(text):
        value = _parse(match.group(0))
        if value >= 100_000 and _DONG.match(text, match.end()):
            value /= 1_000_000
        values.append(value)
    return values
