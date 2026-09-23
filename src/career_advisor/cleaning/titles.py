"""Chuẩn hoá chức danh bằng luật.

Chức danh gốc có 16.720 giá trị vì lẫn mức lương, kinh nghiệm, nơi làm và nhiều vai trò trong
một tên. Chỉ chức danh đủ nhiều tin mới thành node `JobTitle`. Tin còn lại chỉ nối vào nhóm ngành.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from career_advisor.cleaning.locations import tone_new_style

MIN_TITLE_POSTINGS = 20

_BRACKETS = re.compile(r"[(\[].*?[)\]]|[(\[].*$")
# Phần sau các dấu này là mô tả thêm hoặc vai trò thứ hai. Lấy phần đầu.
_SPLIT = re.compile(r"\s+[-–—|]\s+|\s*[,/|]\s*|\s+[–—]|[–—]\s+")
_NOISE_TAILS = (
    re.compile(r"\s+(có\s+)?(hỗ trợ\s+)?(lương|thu nhập|upto|up to)\b.*$"),
    re.compile(r"\s+(làm việc\s+)?tại\s.*$"),
    re.compile(r"\s+(giao tiếp|biết|thành thạo|sử dụng)\s+tiếng\s.*$"),
    re.compile(r"\s+(từ|trên)\s+\d.*$"),
    re.compile(r"\s+(part[\s-]?time|full[\s-]?time|đi làm ngay)\b.*$"),
)
# Không dùng giới tính để gợi ý, nên bỏ khỏi chức danh.
_GENDER = re.compile(r"^(nam|nữ)\s+|\s+(nam|nữ)$")
_GENERIC_PREFIX = re.compile(r"^(nhân viên|nv)\s+(?=\S)")
_SALE = re.compile(r"\bsale\b")  # "sale admin" và "sales admin" là một


def normalize_title(raw: str) -> str:
    text = tone_new_style(unicodedata.normalize("NFC", raw).lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = _BRACKETS.sub(" ", text)
    text = _SPLIT.split(text.strip())[0]
    for pattern in _NOISE_TAILS:
        text = pattern.sub("", text)
    text = _GENDER.sub("", text.strip())
    text = _GENERIC_PREFIX.sub("", text)
    text = _SALE.sub("sales", text)
    return re.sub(r"\s+", " ", text).strip(" .:-")


def add_title_columns(postings: pd.DataFrame, min_postings: int = MIN_TITLE_POSTINGS) -> pd.DataFrame:
    """Thêm 3 cột: `job_title_norm`, `job_title_node` (chỉ khi đủ tin), `job_title_display`.

    `job_title_display` là cách viết gốc phổ biến nhất của chức danh chuẩn, để hiện trên giao diện.
    """
    out = postings.copy()
    out["job_title_norm"] = out["job_title"].map(normalize_title)
    counts = out["job_title_norm"].value_counts()
    frequent = out["job_title_norm"].map(counts) >= min_postings
    out["job_title_node"] = out["job_title_norm"].where(frequent)

    pairs = out.groupby(["job_title_norm", "job_title"]).size().reset_index(name="n")
    pairs = pairs.sort_values(["job_title_norm", "n", "job_title"], ascending=[True, False, True])
    display = pairs.drop_duplicates("job_title_norm").set_index("job_title_norm")["job_title"]
    out["job_title_display"] = out["job_title_norm"].map(display)
    return out
