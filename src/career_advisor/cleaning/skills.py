"""Parse cột kỹ năng và gộp tên trùng bằng luật (tầng 1 của 3 tầng gộp tên).

Cột `technical_skills` và `soft_skills` lưu chuỗi dạng list Python do LLM của nhóm tác giả trích.
Một chuỗi có thể chứa nhiều kỹ năng, ví dụ "Tin học văn phòng (Word, Excel)", và cùng một kỹ năng
có nhiều cách ghi, ví dụ ReactJS, React.js, React 17+. Tầng luật chỉ xử lý các ca chắc chắn.
Các ca mơ hồ để lại cho tầng embedding và tầng LLM.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from functools import cache

import pandas as pd

SOURCES = {"technical": "technical_skills", "soft": "soft_skills"}

# Tiền tố chỉ mức độ, không phải tên kỹ năng. Xếp dài trước ngắn để bóc đúng.
PREFIXES = (
    "sử dụng thành thạo",
    "biết sử dụng",
    "thành thạo",
    "sử dụng",
    "kỹ năng",
    "khả năng",
    "kiến thức về",
    "hiểu biết về",
    "am hiểu",
)
SUFFIXES = ("tốt", "hiệu quả")
# Tên hãng đứng trước tên phần mềm: "adobe photoshop" và "photoshop" là một.
VENDORS = ("adobe", "microsoft", "ms")
# Chỉ các cặp đồng nghĩa chắc chắn. Viết tắt mơ hồ ("ai", "pr", "pp") để tầng LLM quyết.
ALIASES = {
    "office": "tin học văn phòng",
    "vi tính văn phòng": "tin học văn phòng",
    "premiere pro": "premiere",
    "ppt": "powerpoint",
}

_SPACES = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s.,;:]+$")
_PAREN = re.compile(r"^(.*?)\s*\((.*)\)\s*$")
_ITEM_SEP = re.compile(r"\s*[,;]\s*")
# "angular 2+", "python 3", "revit 2020". Không đụng "3ds max" vì số đứng đầu.
_VERSION = re.compile(r"\s+\d+(\.\d+)*\+?$")
# "reactjs", "react js", "react.js" → "react". Cần ít nhất 3 ký tự để không cắt "d3.js".
_JS_SUFFIX = re.compile(r"([a-z][a-z0-9]{2,})[\s.]?js")


def parse_list(raw: object) -> list[str]:
    """Đọc chuỗi dạng list Python. Ô trống trả []. Chuỗi không phải list coi là 1 kỹ năng."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [raw]


def _surface(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _SPACES.sub(" ", text).strip()
    return _TRAILING_PUNCT.sub("", text)


def _is_tool_name(item: str) -> bool:
    # Tên công cụ viết bằng chữ Latin không dấu (Word, Excel). Phần có dấu tiếng Việt
    # như "cao cấp", "nghe, nói" là mô tả mức độ.
    return item.isascii()


def split_compound(raw: str) -> list[str]:
    """Tách một chuỗi thành các kỹ năng con, đã chuẩn hoá bề mặt."""
    text = _surface(raw)
    match = _PAREN.match(text)
    if match:
        head, inner = match.group(1), match.group(2)
        items = [i for i in _ITEM_SEP.split(inner) if i]
        parts = [head, *items] if items and all(_is_tool_name(i) for i in items) else [head]
    elif ":" in text:
        head, rest = text.split(":", 1)
        parts = [head, *_ITEM_SEP.split(rest)]
    else:
        parts = _ITEM_SEP.split(text)
    return [p for p in (_surface(p) for p in parts) if p]


def _strip_words(text: str, words: tuple[str, ...], at_start: bool) -> str:
    changed = True
    while changed:
        changed = False
        for word in words:
            if text == word:
                return ""
            if at_start and text.startswith(word + " "):
                text, changed = text[len(word) + 1 :], True
            elif not at_start and text.endswith(" " + word):
                text, changed = text[: -len(word) - 1], True
    return text


@cache
def canonical_skill(part: str) -> str:
    """Đưa một kỹ năng đã tách về dạng chuẩn. Trả "" nếu không còn gì là tên kỹ năng."""
    text = _surface(part)
    text = _strip_words(text, PREFIXES, at_start=True)
    text = _strip_words(text, SUFFIXES, at_start=False)
    text = _VERSION.sub("", text)
    for vendor in VENDORS:
        if text.startswith(vendor + " "):
            text = text[len(vendor) + 1 :]
            break
    match = _JS_SUFFIX.fullmatch(text)
    if match:
        text = match.group(1)
    text = ALIASES.get(text, text)
    return _SPACES.sub(" ", text).strip()


def classify_kinds(mentions: pd.DataFrame) -> pd.Series:
    """Gán loại cho mỗi kỹ năng: `soft` nếu cột kỹ năng mềm nhắc tới nó ít nhất bằng cột chuyên môn.

    Cách này dựa vào chính dữ liệu thay vì một danh sách tay. Ví dụ "giao tiếp" hay nằm lẫn trong
    cột chuyên môn nhưng xuất hiện nhiều hơn ở cột kỹ năng mềm, nên được xếp lại là `soft`.
    """
    counts = pd.crosstab(mentions["skill"], mentions["source"])
    soft = counts["soft"] if "soft" in counts else 0
    technical = counts["technical"] if "technical" in counts else 0
    kinds = pd.Series("technical", index=counts.index, name="kind")
    kinds[soft >= technical] = "soft"
    return kinds


def build_skill_tables(postings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo 2 bảng từ các tin tuyển dụng.

    - `mentions`: mỗi dòng là một kỹ năng của một tin, đã bỏ trùng trong cùng tin.
      Cột: posting_id, source, raw, skill.
    - `skill_map`: ánh xạ chuỗi gốc → kỹ năng chuẩn. Cột: raw, canonical, kind, tier.
    """
    rows = []
    for source, column in SOURCES.items():
        for posting_id, cell in postings[column].items():
            for raw in parse_list(cell):
                for part in split_compound(raw):
                    skill = canonical_skill(part)
                    if skill:
                        rows.append((posting_id, source, raw, skill))
    all_mentions = pd.DataFrame(rows, columns=["posting_id", "source", "raw", "skill"])

    kinds = classify_kinds(all_mentions)
    skill_map = (
        all_mentions[["raw", "skill"]]
        .drop_duplicates()
        .rename(columns={"skill": "canonical"})
        .assign(kind=lambda d: d["canonical"].map(kinds), tier="rule")
        .reset_index(drop=True)
    )
    mentions = all_mentions.drop_duplicates(["posting_id", "skill"]).reset_index(drop=True)
    return mentions, skill_map
