import math

import pandas as pd
import pytest

from career_advisor import data
from career_advisor.cleaning.fields import (
    clean_postings,
    education_level,
    experience_level,
    experience_months,
    parse_salary,
    salary_band,
)
from career_advisor.cleaning.locations import FOREIGN


@pytest.mark.parametrize(
    "text, salary_max, expected",
    [
        ("10.0 - 15.0 triệu", 15, (10.0, 15.0)),
        # Cột salary_min bị cắt thành số nguyên (0,5 → 0). Lấy số từ chuỗi gốc.
        ("0.5 - 1.0 triệu", 1, (0.5, 1.0)),
        ("18.0 triệu", 18, (18.0, 18.0)),
    ],
)
def test_parse_salary_reads_decimals_from_the_text(text, salary_max, expected):
    assert parse_salary(text, salary_max) == expected


@pytest.mark.parametrize("text", ["0.0 triệu", "8.0 - 0.0 triệu"])
def test_negotiable_salary_is_missing(text):
    low, high = parse_salary(text, 0)

    assert math.isnan(low) and math.isnan(high)


@pytest.mark.parametrize(
    "mid, band",
    [(5, "under_7m"), (7, "7_10m"), (12.5, "10_15m"), (15, "15_20m"), (25, "20_30m"), (49.9, "30_50m"),
     (50, "50m_plus"), (float("nan"), None)],
)  # fmt: skip
def test_salary_band(mid, band):
    assert salary_band(mid) == band


@pytest.mark.parametrize(
    "text, months",
    [("Không yêu cầu", 0), ("6 tháng", 6), ("1 năm", 12), ("18 tháng", 18), ("5 năm", 60)],
)
def test_experience_months(text, months):
    assert experience_months(text) == months


def test_experience_months_missing_is_nan():
    assert math.isnan(experience_months(None))


@pytest.mark.parametrize(
    "months, level",
    [(0, "none"), (6, "under_1y"), (12, "1_2y"), (24, "1_2y"), (30, "3_4y"), (48, "3_4y"), (60, "5y_plus")],
)
def test_experience_level(months, level):
    assert experience_level(months) == level


@pytest.mark.parametrize(
    "items, level",
    [
        (["Cao Đẳng trở lên"], "college"),
        (["Trung học phổ thông (Cấp 3) trở lên"], "high_school"),
        (["Tốt nghiệp THPT"], "high_school"),
        (["Trung học cơ sở (Cấp 2) trở lên"], "lower_secondary"),
        (["Trung cấp trở lên"], "vocational"),
        (["Đại học", "Kế toán"], "university"),
        (["Thạc sĩ"], "postgraduate"),
        (["Sau đại học"], "postgraduate"),
        # Tin ghi nhiều bậc: lấy bậc thấp nhất, vì đó là yêu cầu tối thiểu.
        (["Đại học", "Cao đẳng"], "college"),
        (["Không yêu cầu bằng cấp"], "none"),
        (["Bachelor's degree in Computer Science or related field"], "university"),
        (["Tốt nghiệp CĐ/ĐH trở lên"], "college"),
        (["Tốt nghiệp ĐH"], "university"),
        (["High school diploma"], "high_school"),
        (["Marketing"], "unspecified"),
        ([], "unspecified"),
    ],
)
def test_education_level(items, level):
    assert education_level(items) == level


def _raw_postings() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "job_title": ["Kế toán", "Dev", "Sales"],
            "location": ["hà đông", "hà nội, hồ chí minh", "hồ chí minh"],
            "country": ["Việt Nam", "Việt Nam", "Mỹ"],
            "qualifications": ["['Cao Đẳng trở lên', 'Kế toán']", "['Đại học']", None],
            "experience_required": ["1 năm", "Không yêu cầu", "3 năm"],
            "salary": ["10.0 - 15.0 triệu", "0.0 triệu", "20.0 triệu"],
            "salary_max": [15, 0, 20],
            "languages_required": ["tiếng anh,tiếng trung", None, "UNK"],
            "contract_type": ["Toàn thời gian"] * 3,
            "category": ["tài_chính", "cntt", "kinh_doanh"],
            "description": ["a", "b", "c"],
            "requirements_text": ["x", "y", "z"],
        }
    )
    df.index.name = "posting_id"
    return df


def test_clean_postings_builds_one_row_per_posting_with_clean_fields():
    clean = clean_postings(_raw_postings())

    assert clean.index.equals(_raw_postings().index)
    assert clean.loc[0, "provinces"] == ["hà nội"]
    assert clean.loc[1, "provinces"] == ["hà nội", "hồ chí minh"]
    assert clean.loc[0, "salary_mid"] == 12.5
    assert clean.loc[0, "salary_band"] == "10_15m"
    assert clean.loc[0, "experience_level"] == "1_2y"
    assert clean.loc[0, "education_level"] == "college"
    assert clean.loc[0, "majors"] == ["kế toán"]
    assert clean.loc[0, "languages"] == ["tiếng anh", "tiếng trung"]


def test_clean_postings_marks_negotiable_salary():
    clean = clean_postings(_raw_postings())

    assert bool(clean.loc[1, "salary_negotiable"])
    assert pd.isna(clean.loc[1, "salary_mid"]) and pd.isna(clean.loc[1, "salary_band"])


def test_posting_outside_vietnam_is_foreign():
    clean = clean_postings(_raw_postings())

    assert clean.loc[2, "provinces"] == [FOREIGN]
    assert clean.loc[2, "languages"] == []


@pytest.mark.data
@pytest.mark.skipif(not data.RAW_CSV.exists(), reason="chưa tải dữ liệu")
def test_real_data_has_13732_negotiable_salaries():
    clean = clean_postings(data.load_postings())

    assert clean["salary_negotiable"].sum() == 13_732
    assert clean["salary_mid"].isna().sum() == 13_732
