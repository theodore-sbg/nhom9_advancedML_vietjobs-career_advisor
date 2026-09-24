"""Chuẩn hoá địa điểm về 34 tỉnh thành sau sáp nhập (hiệu lực 01/07/2025).

Dữ liệu đăng từ tháng 7 đến tháng 10/2025, nên tin trộn cả tên tỉnh mới, tên tỉnh cũ và tên
quận, huyện ("hà đông", "quận 7"). Mọi tên được quy về tỉnh mới. Tin ghi nhiều nơi thì giữ
danh sách nhiều tỉnh.
"""

from __future__ import annotations

import re
import unicodedata

UNKNOWN = "unknown"
FOREIGN = "foreign"

# 11 tỉnh giữ nguyên và 23 tỉnh mới sau sáp nhập. Dấu thanh đặt kiểu mới ("hoà", "hoá").
PROVINCES: tuple[str, ...] = (
    "hà nội", "huế", "lai châu", "điện biên", "sơn la", "lạng sơn", "quảng ninh", "thanh hoá",
    "nghệ an", "hà tĩnh", "cao bằng",
    "tuyên quang", "lào cai", "thái nguyên", "phú thọ", "bắc ninh", "hưng yên", "hải phòng",
    "ninh bình", "quảng trị", "đà nẵng", "quảng ngãi", "gia lai", "khánh hoà", "lâm đồng",
    "đắk lắk", "hồ chí minh", "đồng nai", "tây ninh", "cần thơ", "vĩnh long", "đồng tháp",
    "cà mau", "an giang",
)  # fmt: skip

# Tỉnh cũ → tỉnh mới, theo Nghị quyết 202/2025/QH15.
_OLD_PROVINCES = {
    "hà giang": "tuyên quang",
    "yên bái": "lào cai",
    "bắc kạn": "thái nguyên",
    "vĩnh phúc": "phú thọ",
    "hoà bình": "phú thọ",
    "bắc giang": "bắc ninh",
    "thái bình": "hưng yên",
    "hải dương": "hải phòng",
    "hà nam": "ninh bình",
    "nam định": "ninh bình",
    "quảng bình": "quảng trị",
    "thừa thiên huế": "huế",
    "quảng nam": "đà nẵng",
    "kon tum": "quảng ngãi",
    "bình định": "gia lai",
    "ninh thuận": "khánh hoà",
    "phú yên": "đắk lắk",
    "đắk nông": "lâm đồng",
    "bình thuận": "lâm đồng",
    "bình phước": "đồng nai",
    "bình dương": "hồ chí minh",
    "bà rịa vũng tàu": "hồ chí minh",
    "long an": "tây ninh",
    "tiền giang": "đồng tháp",
    "bến tre": "vĩnh long",
    "trà vinh": "vĩnh long",
    "sóc trăng": "cần thơ",
    "hậu giang": "cần thơ",
    "bạc liêu": "cà mau",
    "kiên giang": "an giang",
}

# Quận, huyện, thành phố trực thuộc, và cách viết khác hay gặp trong dữ liệu.
_PLACES = {
    "hà nội": (
        "ha noi",
        "hanoi",
        "hn",
        "hà đông",
        "bắc từ liêm",
        "nam từ liêm",
        "từ liêm",
        "cầu giấy",
        "đống đa",
        "long biên",
        "thanh xuân",
        "hoàng mai",
        "hai bà trưng",
        "ba đình",
        "hoàn kiếm",
        "tây hồ",
        "đông anh",
        "gia lâm",
        "chương mỹ",
        "thường tín",
        "quốc oai",
        "đan phượng",
        "hoài đức",
        "sơn tây",
        "thanh trì",
        "mê linh",
        "sóc sơn",
        "thạch thất",
        "phúc thọ",
        "ba vì",
        "mỹ đức",
        "ứng hoà",
        "phú xuyên",
        "thanh oai",
        "mỹ đình",
        "linh đàm",
        "đại kim",
        "vĩnh tuy",
    ),
    "hồ chí minh": (
        "hcm",
        "tphcm",
        "tp hcm",
        "ho chi minh",
        "ho chi minh city",
        "sài gòn",
        "sai gon",
        "thủ đức",
        "bình thạnh",
        "nhà bè",
        "củ chi",
        "tân bình",
        "hóc môn",
        "bình tân",
        "gò vấp",
        "tân phú",
        "phú nhuận",
        "bình chánh",
        "cần giờ",
        "bến cát",
        "dĩ an",
        "thủ dầu một",
        "thuận an",
        "tân uyên",
        "bàu bàng",
        "vũng tàu",
        "bà rịa",
        "phú mỹ",
        "long điền",
        "châu đức",
        "brvt",
    ),
    "nghệ an": ("vinh", "cửa lò", "nghi lộc", "hưng nguyên", "nghĩa đàn"),
    "khánh hoà": ("nha trang", "cam ranh", "phan rang", "phan rang tháp chàm"),
    "đắk lắk": ("đắc lắc", "đắk lắc", "dak lak", "buôn ma thuột", "buôn mê thuột", "tuy hoà", "cư kuin"),
    "đồng nai": ("biên hoà", "long khánh", "long thành", "nhơn trạch", "trảng bom", "đồng xoài"),
    "lâm đồng": ("đắc nông", "dak nong", "đà lạt", "bảo lộc", "đơn dương", "đức trọng", "phan thiết"),
    "an giang": ("phú quốc", "rạch giá", "long xuyên", "châu đốc"),
    "quảng ninh": ("hạ long", "cẩm phả", "móng cái", "uông bí", "đông triều", "quảng yên"),
    "ninh bình": ("phủ lý", "hoa lư", "tam điệp", "duy tiên"),
    "gia lai": ("quy nhơn", "pleiku", "an nhơn"),
    "hưng yên": ("mỹ hào", "yên mỹ", "văn lâm", "văn giang", "khoái châu"),
    "đà nẵng": (
        "điện bàn",
        "hội an",
        "tam kỳ",
        "hải châu",
        "sơn trà",
        "liên chiểu",
        "ngũ hành sơn",
        "cẩm lệ",
        "núi thành",
        "thăng bình",
        "duy xuyên",
    ),
    "bắc ninh": ("từ sơn", "yên phong", "quế võ", "thuận thành", "tiên du", "việt yên"),
    "hải phòng": (
        "cẩm giàng",
        "chí linh",
        "kinh môn",
        "thủy nguyên",
        "an dương",
        "kiến an",
        "hồng bàng",
        "lê chân",
        "hải an",
        "tiên lãng",
    ),
    "thanh hoá": ("bỉm sơn", "sầm sơn", "nghi sơn", "ngọc lặc"),
    "đồng tháp": ("mỹ tho", "cao lãnh", "sa đéc"),
    "tây ninh": ("bến lức", "tân an", "đức hoà", "cần giuộc", "cần đước", "trảng bàng"),
    "phú thọ": ("bình xuyên", "việt trì", "vĩnh yên", "phúc yên", "tam đảo"),
    "cần thơ": ("ninh kiều", "cái răng", "vị thanh"),
    "huế": ("hue",),
    "quảng ngãi": ("dung quất",),
    "quảng trị": ("đồng hới",),
    "điện biên": ("điện biên phủ",),
    "thái nguyên": ("sông công", "phổ yên"),
    "lào cai": ("sa pa", "sapa"),
}

_FOREIGN_PLACES = (
    "nước ngoài", "mỹ", "hoa kỳ", "nhật bản", "hàn quốc", "trung quốc", "đài loan", "singapore",
    "úc", "đức", "canada", "san francisco", "new york", "tokyo", "osaka", "seattle",
    "mountain view", "mozambique",
)  # fmt: skip

_TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng
# "hòa" → "hoà", "khỏe" → "khoẻ", "thủy" → "thuỷ": đưa dấu thanh về kiểu mới.
_OLD_TONE = re.compile(f"(o)([{_TONE_MARKS}])([ae])|(u)([{_TONE_MARKS}])(y)")
_SEPARATORS = re.compile(r"\s*(?:,|;|/|&|\(|\)|\s-\s|\bvà\b|\bhoặc\b)\s*")
_PREFIXES = re.compile(r"^(?:thành phố|tp\.?|tỉnh|thị xã|tx\.?|huyện|quận|q\.)\s*")
_NUMBERED_DISTRICT = re.compile(r"(?:quận|q\.?|district)\s*\d+")


def tone_new_style(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    nfd = _OLD_TONE.sub(lambda m: "".join(g for g in (m[1], m[3], m[2], m[4], m[6], m[5]) if g), nfd)
    return unicodedata.normalize("NFC", nfd)


def _key(text: str) -> str:
    text = tone_new_style(text.lower())
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip(" .")


def _build_lookup() -> dict[str, str]:
    lookup = {_key(p): p for p in PROVINCES}
    lookup.update({_key(old): new for old, new in _OLD_PROVINCES.items()})
    for province, places in _PLACES.items():
        lookup.update({_key(place): province for place in places})
    lookup.update({_key(place): FOREIGN for place in _FOREIGN_PLACES})
    return lookup


PLACE_TO_PROVINCE = _build_lookup()


def province_of(token: str) -> str | None:
    key = _key(token)
    if key in PLACE_TO_PROVINCE:
        return PLACE_TO_PROVINCE[key]
    if _NUMBERED_DISTRICT.fullmatch(key):
        return "hồ chí minh"  # chỉ TP.HCM có quận đánh số
    stripped = _PREFIXES.sub("", key)
    return PLACE_TO_PROVINCE.get(stripped)


def normalize_location(raw: str | None) -> list[str]:
    """Trả về các tỉnh (theo thứ tự xuất hiện, không trùng) trong chuỗi địa điểm.

    Không nhận ra nơi nào thì trả [UNKNOWN]. Nơi ở nước ngoài trả FOREIGN.
    """
    if not isinstance(raw, str):
        return [UNKNOWN]
    text = re.sub(r"[\[\]'\"]", "", raw)
    found: list[str] = []
    for token in _SEPARATORS.split(text):
        province = province_of(token) if token.strip() else None
        if province and province not in found:
            found.append(province)
    return found or [UNKNOWN]
