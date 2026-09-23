"""Chuẩn hoá lương, kinh nghiệm, bằng cấp, ngôn ngữ và gộp thành bảng tin sạch."""

from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd

from career_advisor.cleaning.locations import FOREIGN, normalize_location
from career_advisor.cleaning.skills import parse_list

VIETNAM = "Việt Nam"

# Khoảng lương theo mức giữa, đơn vị triệu đồng/tháng. Cận dưới thuộc khoảng.
SALARY_BANDS = (
    (7, "under_7m"),
    (10, "7_10m"),
    (15, "10_15m"),
    (20, "15_20m"),
    (30, "20_30m"),
    (50, "30_50m"),
    (math.inf, "50m_plus"),
)

# Mức kinh nghiệm theo số tháng, cận trên thuộc khoảng.
EXPERIENCE_LEVELS = ((0, "none"), (11, "under_1y"), (24, "1_2y"), (48, "3_4y"), (math.inf, "5y_plus"))

# Bậc học từ thấp tới cao. "sau đại học" phải xét trước "đại học".
EDUCATION_ORDER = (
    "none",
    "lower_secondary",
    "high_school",
    "vocational",
    "college",
    "university",
    "postgraduate",
)
_EDUCATION_PATTERNS = (
    ("none", r"không yêu cầu"),
    ("postgraduate", r"thạc sĩ|tiến sĩ|sau đại học|master|phd"),
    ("lower_secondary", r"trung học cơ sở|cấp 2|thcs"),
    ("high_school", r"trung học phổ thông|cấp 3|thpt|phổ thông|high school"),
    ("vocational", r"trung cấp"),
    ("college", r"cao đẳng|\bcđ\b"),
    ("university", r"đại học|\bđh\b|cử nhân|kỹ sư|bachelor"),
)

_RANGE = re.compile(r"([\d.]+)\s*-\s*([\d.]+)\s*triệu")
_SINGLE = re.compile(r"([\d.]+)\s*triệu")
_EXPERIENCE = re.compile(r"(\d+)\s*(năm|tháng)")


def parse_salary(text: str | None, salary_max: float) -> tuple[float, float]:
    """Trả (min, max) theo triệu đồng. Lương thoả thuận (`salary_max == 0`) trả (nan, nan).

    Số lấy từ chuỗi gốc vì cột `salary_min` của bộ dữ liệu bị cắt thành số nguyên.
    """
    if not salary_max or not isinstance(text, str):
        return (math.nan, math.nan)
    if match := _RANGE.search(text):
        return (float(match[1]), float(match[2]))
    if match := _SINGLE.search(text):
        return (float(match[1]), float(match[1]))
    return (math.nan, math.nan)


def salary_band(mid: float) -> str | None:
    if mid is None or math.isnan(mid):
        return None
    return next(label for upper, label in SALARY_BANDS if mid < upper)


def experience_months(text: str | None) -> float:
    if not isinstance(text, str):
        return math.nan
    if "không yêu cầu" in text.lower():
        return 0
    match = _EXPERIENCE.search(text.lower())
    if not match:
        return math.nan
    return int(match[1]) * (12 if match[2] == "năm" else 1)


def experience_level(months: float) -> str | None:
    if months is None or math.isnan(months):
        return None
    return next(label for upper, label in EXPERIENCE_LEVELS if months <= upper)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def _item_level(item: str) -> str | None:
    text = _norm(item)
    return next((level for level, pattern in _EDUCATION_PATTERNS if re.search(pattern, text)), None)


def education_level(items: list[str]) -> str:
    """Bậc học thấp nhất được nhắc tới, vì đó là yêu cầu tối thiểu. Không có thì `unspecified`."""
    levels = [level for level in map(_item_level, items) if level]
    if not levels:
        return "unspecified"
    return min(levels, key=EDUCATION_ORDER.index)


def majors(items: list[str]) -> list[str]:
    """Các mục không chứa bậc học, ví dụ "Kế toán", "Marketing", được coi là chuyên ngành."""
    return [_norm(i) for i in items if _item_level(i) is None]


def languages(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return [lang for lang in (_norm(t) for t in text.split(",")) if lang and lang != "unk"]


_PASSTHROUGH = ("job_title", "category", "contract_type", "description", "requirements_text", "benefits")


def clean_postings(raw: pd.DataFrame) -> pd.DataFrame:
    """Bảng tin sạch, cùng index `posting_id` với dữ liệu gốc."""
    out = raw[[c for c in _PASSTHROUGH if c in raw]].copy()

    abroad = raw["country"].fillna(VIETNAM) != VIETNAM
    out["provinces"] = [
        [FOREIGN] if is_abroad else normalize_location(loc)
        for loc, is_abroad in zip(raw["location"], abroad, strict=True)
    ]

    salary = [parse_salary(t, m) for t, m in zip(raw["salary"], raw["salary_max"], strict=True)]
    out["salary_min"] = [low for low, _ in salary]
    out["salary_max"] = [high for _, high in salary]
    out["salary_mid"] = (out["salary_min"] + out["salary_max"]) / 2
    out["salary_negotiable"] = out["salary_mid"].isna()
    out["salary_band"] = out["salary_mid"].map(salary_band)

    out["experience_months"] = raw["experience_required"].map(experience_months)
    out["experience_level"] = out["experience_months"].map(experience_level)

    qualifications = raw["qualifications"].map(parse_list)
    out["education_level"] = qualifications.map(education_level)
    out["majors"] = qualifications.map(majors)
    out["languages"] = raw["languages_required"].map(languages)
    return out
