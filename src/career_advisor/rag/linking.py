"""Nối thực thể trong câu hỏi vào node của đồ thị: chức danh, nhóm ngành, kỹ năng, tỉnh, kinh nghiệm.

Thứ tự: nhóm ngành → chức danh → kỹ năng. Cụm đã dùng cho chức danh hay nhóm ngành không được đọc lại
thành kỹ năng ("kế toán tổng hợp" trong "muốn làm kế toán tổng hợp" là nghề, không phải kỹ năng).
Ngoài ra đánh dấu tín hiệu ngoài phạm vi: nơi ở nước ngoài, năm khác năm của dữ liệu.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import networkx as nx

from career_advisor.cleaning.fields import experience_level
from career_advisor.cleaning.locations import FOREIGN, UNKNOWN, province_of, tone_new_style
from career_advisor.cleaning.titles import normalize_title
from career_advisor.retrieval.hybrid import extract_skills

DATA_YEAR = 2025
MAX_TITLE_WORDS = 6
MAX_PLACE_WORDS = 4

# Cụm từ của chính câu hỏi. Dữ liệu có node kỹ năng rác trùng chữ ("trung", "nhóm"), nên các cụm này
# được đánh dấu là đã dùng trước khi tìm kỹ năng.
QUESTION_PHRASES = (
    "lương trung vị", "mức lương", "lương", "trung vị", "nhóm ngành", "ngành", "chức danh", "bao nhiêu",
    "kinh nghiệm", "tin tuyển dụng", "tin tuyển", "tuyển dụng", "kỹ năng", "còn thiếu", "quan trọng nhất",
    "phần trăm", "hay đi cùng", "muốn làm", "tôi biết", "cao nhất", "khoảng",
)  # fmt: skip

_WORD = re.compile(r"\w+")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_EXPERIENCE = (
    (re.compile(r"không yêu cầu kinh nghiệm|chưa có kinh nghiệm"), lambda m: 0),
    (re.compile(r"dưới (\d+) năm"), lambda m: int(m[1]) * 12 - 1),
    (re.compile(r"(?:từ|trên) (\d+) năm"), lambda m: int(m[1]) * 12),
    (re.compile(r"(\d+)\s*[–-]\s*\d+ năm"), lambda m: int(m[1]) * 12),
    (re.compile(r"(\d+) năm kinh nghiệm|(\d+) năm làm"), lambda m: int(m[1] or m[2]) * 12),
    (re.compile(r"(\d+) tháng kinh nghiệm"), lambda m: int(m[1])),
)


@dataclass
class LinkedEntities:
    skills: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    provinces: list[str] = field(default_factory=list)
    experience_levels: list[str] = field(default_factory=list)
    foreign: bool = False
    other_years: list[int] = field(default_factory=list)


def _normalize(text: str) -> str:
    return tone_new_style(unicodedata.normalize("NFC", text).lower())


def _names(G: nx.DiGraph, node_type: str) -> set[str]:
    return {d["name"] for _, d in G.nodes(data=True) if d.get("type") == node_type}


def _match_spans(words: list[str], used: list[bool], max_words: int, lookup) -> list[tuple[int, int, str]]:
    """Khớp tham lam cụm dài nhất trước, không chồng lên phần đã dùng."""
    found = []
    for n in range(min(max_words, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            if any(used[i : i + n]):
                continue
            value = lookup(" ".join(words[i : i + n]))
            if value:
                found.append((i, i + n, value))
                used[i : i + n] = [True] * n
    return sorted(found)


def _mark(words: list[str], used: list[bool], phrase: str) -> None:
    """Đánh dấu mọi chỗ cụm `phrase` xuất hiện trong `words` là đã dùng."""
    target = phrase.split()
    for i in range(len(words) - len(target) + 1):
        if words[i : i + len(target)] == target:
            used[i : i + len(target)] = [True] * len(target)


def _segments(words: list[str], used: list[bool]) -> list[str]:
    """Các đoạn chưa dùng, tách rời nhau để không ghép chữ qua chỗ trống."""
    parts, current = [], []
    for word, is_used in zip(words, used, strict=True):
        if is_used:
            if current:
                parts.append(" ".join(current))
            current = []
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def link_entities(G: nx.DiGraph, text: str) -> LinkedEntities:
    norm = _normalize(text)
    words = _WORD.findall(norm)
    used = [False] * len(words)
    linked = LinkedEntities()

    categories = _names(G, "Category")
    pretty = {c.replace("_", " "): c for c in categories}
    for phrase, category in sorted(pretty.items(), key=lambda item: -len(item[0])):
        head = " ".join(phrase.split()[:3])
        matched = phrase if phrase in norm else head if f"ngành {head}" in norm else None
        if matched:
            linked.categories.append(category)
            _mark(words, used, matched)

    titles = _names(G, "JobTitle")
    # Chức danh xét trước cụm từ câu hỏi: "kinh doanh" không bị che bởi "ngành".
    title_spans = _match_spans(
        words, used, MAX_TITLE_WORDS, lambda s: normalize_title(s) if normalize_title(s) in titles else None
    )
    linked.titles = list(dict.fromkeys(t for _, _, t in title_spans))

    for phrase in sorted(QUESTION_PHRASES, key=lambda p: -len(p)):
        _mark(words, used, phrase)

    places = []
    place_used = [False] * len(words)
    for _, _, province in _match_spans(words, place_used, MAX_PLACE_WORDS, province_of):
        if province == FOREIGN:
            linked.foreign = True
        elif province != UNKNOWN:
            places.append(province)
    linked.provinces = list(dict.fromkeys(places))
    used = [u or p for u, p in zip(used, place_used, strict=True)]

    skills: set[str] = set()
    for segment in _segments(words, used):
        skills |= extract_skills(G, segment)
    linked.skills = sorted(skills - set(linked.titles))

    for pattern, months in _EXPERIENCE:
        match = pattern.search(norm)
        if match:
            linked.experience_levels = [experience_level(months(match))]
            break

    linked.other_years = sorted({int(y) for y in _YEAR.findall(norm)} - {DATA_YEAR})
    return linked
