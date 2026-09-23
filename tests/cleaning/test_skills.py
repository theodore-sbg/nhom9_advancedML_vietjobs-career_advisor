import pandas as pd
import pytest

from career_advisor.cleaning.skills import (
    build_skill_tables,
    canonical_skill,
    classify_kinds,
    parse_list,
    split_compound,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("['Excel', 'Word']", ["Excel", "Word"]),
        ("[]", []),
        (None, []),
        (float("nan"), []),
        ("Excel", ["Excel"]),  # không phải list thì coi là 1 kỹ năng
    ],
)
def test_parse_list(raw, expected):
    assert parse_list(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Tin học văn phòng (Word, Excel, PowerPoint)", ["tin học văn phòng", "word", "excel", "powerpoint"]),
        ("Tin học văn phòng: Word, Excel", ["tin học văn phòng", "word", "excel"]),
        (
            "Framework JavaScript (Angular 2+, ReactJS, VueJS)",
            ["framework javascript", "angular 2+", "reactjs", "vuejs"],
        ),
        # Phần trong ngoặc là mô tả mức độ, không phải công cụ: bỏ đi.
        ("Tiếng Trung (nghe, nói, đọc, viết)", ["tiếng trung"]),
        ("Excel (cao cấp)", ["excel"]),
        # Dấu "/" thường là một khái niệm liền (CI/CD), nên không tách.
        ("CI/CD", ["ci/cd"]),
        ("Word, Excel; PowerPoint", ["word", "excel", "powerpoint"]),
        ("  Photoshop.  ", ["photoshop"]),
    ],
)
def test_split_compound(raw, expected):
    assert split_compound(raw) == expected


@pytest.mark.parametrize(
    "variant", ["ReactJS", "React", "React 17+", "ReactJs", "React JS", "reactjs", "React.js", "react.JS"]
)
def test_react_variants_collapse_to_one_skill(variant):
    assert canonical_skill(variant.lower()) == "react"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("react native", "react native"),  # kỹ năng khác, không gộp vào react
        ("node.js", "node"),
        ("vuejs", "vue"),
        ("javascript", "javascript"),
        ("adobe photoshop", "photoshop"),
        ("adobe premiere", "premiere"),
        ("microsoft excel", "excel"),
        ("ms excel", "excel"),
        ("microsoft office", "tin học văn phòng"),
        ("thành thạo tin học văn phòng", "tin học văn phòng"),
        ("sử dụng thành thạo tin học văn phòng", "tin học văn phòng"),
        ("kỹ năng tin học văn phòng", "tin học văn phòng"),
        ("kỹ năng giao tiếp", "giao tiếp"),
        ("giao tiếp tốt", "giao tiếp"),
        ("khả năng làm việc nhóm", "làm việc nhóm"),
        ("angular 2+", "angular"),
        ("python 3", "python"),
        ("c#", "c#"),
        ("c++", "c++"),
        (".net", ".net"),
        ("3ds max", "3ds max"),
        ("kỹ năng", ""),
        ("vi tính văn phòng", "tin học văn phòng"),
        ("thành thạo vi tính văn phòng", "tin học văn phòng"),
        ("adobe premiere pro", "premiere"),
        ("ppt", "powerpoint"),
        # Viết tắt mơ hồ: để tầng LLM quyết theo ngữ cảnh.
        ("ai", "ai"),
        ("pr", "pr"),
        ("pp", "pp"),
    ],
)
def test_canonical_skill(raw, expected):
    assert canonical_skill(raw) == expected


def test_canonical_skill_normalizes_unicode_to_nfc():
    decomposed = "giao tiếp"  # "giao tiếp" dạng NFD

    assert canonical_skill(decomposed) == "giao tiếp"


def test_classify_kinds_uses_which_column_mentions_the_skill_more():
    mentions = pd.DataFrame(
        {
            "skill": ["giao tiếp"] * 3 + ["excel"] * 3 + ["đàm phán"] * 2,
            "source": ["soft", "soft", "technical", "technical", "technical", "soft", "soft", "technical"],
        }
    )

    kinds = classify_kinds(mentions)

    assert kinds.to_dict() == {"giao tiếp": "soft", "excel": "technical", "đàm phán": "soft"}


def _postings() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "technical_skills": [
                "['ReactJS', 'React.js', 'Kỹ năng giao tiếp']",
                "['Tin học văn phòng (Word, Excel)']",
                None,
            ],
            "soft_skills": ["['Giao tiếp tốt']", "[]", "['Giao tiếp']"],
        }
    )
    df.index.name = "posting_id"
    return df


def test_build_skill_tables_dedupes_skills_within_a_posting():
    mentions, _ = build_skill_tables(_postings())

    first = mentions[mentions.posting_id == 0]
    assert sorted(first.skill) == ["giao tiếp", "react"]


def test_build_skill_tables_moves_soft_skill_out_of_technical_kind():
    mentions, skill_map = build_skill_tables(_postings())

    kinds = skill_map.drop_duplicates("canonical").set_index("canonical")["kind"]
    assert kinds["giao tiếp"] == "soft"
    assert kinds["react"] == "technical"


def test_skill_map_links_every_raw_string_to_its_canonical_skills():
    _, skill_map = build_skill_tables(_postings())

    office = skill_map[skill_map.raw == "Tin học văn phòng (Word, Excel)"]
    assert sorted(office.canonical) == ["excel", "tin học văn phòng", "word"]
    assert set(skill_map.tier) == {"rule"}
    assert list(skill_map.columns) == ["raw", "canonical", "kind", "tier"]


def test_mentions_keep_posting_id_and_source_column():
    mentions, _ = build_skill_tables(_postings())

    assert list(mentions.columns) == ["posting_id", "source", "raw", "skill"]
    assert set(mentions[mentions.posting_id == 2].source) == {"soft"}
