import pytest

from career_advisor.rag.numbers import extract_numbers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Lương trung vị 14,5 triệu", [14.5]),
        ("khoảng 12.5 triệu", [12.5]),  # LLM đôi khi dùng dấu chấm thập phân
        ("tính trên 1.178 tin", [1178]),  # dấu chấm phân cách hàng nghìn
        ("12.500.000 đồng", [12.5]),  # đổi đồng → triệu
        ("12tr5", [12.5]),
        ("15 tr", [15]),
        ("50,9% tin", [50.9]),
        ("từ 13,5–18 triệu", [13.5, 18]),
        ("mã tin [#1234]", []),  # mã tin không phải số liệu
        ("năm 2025, 3 năm kinh nghiệm", [3]),  # năm của dữ liệu không phải số liệu
        ("không có số", []),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)
