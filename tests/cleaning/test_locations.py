import pytest

from career_advisor.cleaning.locations import FOREIGN, PROVINCES, UNKNOWN, normalize_location


def test_there_are_34_provinces_after_the_2025_merger():
    assert len(PROVINCES) == 34


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hà nội", ["hà nội"]),
        ("hồ chí minh", ["hồ chí minh"]),
        # Quận, huyện, thành phố trực thuộc → tỉnh.
        ("hà đông", ["hà nội"]),
        ("quận 7", ["hồ chí minh"]),
        ("thủ đức", ["hồ chí minh"]),
        ("tp vinh", ["nghệ an"]),
        ("tp. nha trang", ["khánh hoà"]),
        # Tỉnh cũ trước sáp nhập → tỉnh mới.
        ("bắc giang", ["bắc ninh"]),
        ("bình dương", ["hồ chí minh"]),
        ("hải dương", ["hải phòng"]),
        ("quảng nam", ["đà nẵng"]),
        ("thừa thiên huế", ["huế"]),
        # Viết tắt, sai chính tả thường gặp.
        ("tp hcm", ["hồ chí minh"]),
        ("đắc nông", ["lâm đồng"]),
        # Dấu thanh đặt ở 2 kiểu: "hòa" và "hoà".
        ("biên hòa", ["đồng nai"]),
        ("biên hoà", ["đồng nai"]),
        ("thanh hóa", ["thanh hoá"]),
        # Nhiều nơi trong một tin: giữ thứ tự, bỏ trùng.
        ("hà nội, hồ chí minh", ["hà nội", "hồ chí minh"]),
        ("hà nội, hà đông", ["hà nội"]),
        ("['hà nội', 'hồ chí minh']", ["hà nội", "hồ chí minh"]),
        ("Hà Nội - Hồ Chí Minh", ["hà nội", "hồ chí minh"]),
        ("thủ đức hoặc dĩ an", ["hồ chí minh"]),
        ("district 7", ["hồ chí minh"]),
        ("ho chi minh city", ["hồ chí minh"]),
        ("tp đồng hới", ["quảng trị"]),
        ("hải phòng (cẩm giàng)", ["hải phòng"]),
        ("seattle", [FOREIGN]),
        # Nước ngoài và không nhận ra.
        ("san francisco", [FOREIGN]),
        ("nước ngoài", [FOREIGN]),
        ("xyz abc", [UNKNOWN]),
        ("", [UNKNOWN]),
        (None, [UNKNOWN]),
    ],
)
def test_normalize_location(raw, expected):
    assert normalize_location(raw) == expected


def test_known_places_are_never_mixed_with_unknown():
    assert normalize_location("hà nội, xyz abc") == ["hà nội"]


def test_every_mapped_place_is_one_of_the_34_provinces():
    from career_advisor.cleaning.locations import PLACE_TO_PROVINCE

    assert set(PLACE_TO_PROVINCE.values()) <= set(PROVINCES) | {FOREIGN}
