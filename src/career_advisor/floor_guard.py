"""Kiểm phần nền của CONSTRAINTS.md trên diff: chỉ bắt những thay đổi làm hạ chuẩn.

Năm kiểu hạ chuẩn: thêm chú thích tắt kiểm tra, làm test dễ đi (skip, bỏ assert), để code dở dang,
lộ bí mật, và nới CONSTRAINTS.md (hạ con số, thêm ngoại lệ). Báo luật và vị trí, không in giá trị bí mật.
"""

import re
from collections import Counter
from dataclasses import dataclass

# Tệp định nghĩa chính các mẫu này: miễn kiểm dòng thêm để không tự bắt chính mình.
# Không miễn kiểm dòng xoá: bỏ assert khỏi test của bộ kiểm vẫn phải bị báo.
EXEMPT = {"src/career_advisor/floor_guard.py", "tests/test_floor_guard.py"}

SUPPRESSIONS = re.compile(
    r"#\s*noqa|#\s*type:\s*ignore|#\s*pragma:\s*no cover|#\s*nosec|nosemgrep|gitleaks:allow"
)
SKIPS = re.compile(r"@pytest\.mark\.(skip|xfail)\b|pytest\.skip\(|unittest\.skip")
STUBS = re.compile(r"raise NotImplementedError|\bTODO\b|\bFIXME\b|except[^:]*:\s*(pass|\.\.\.)\s*$")
SECRETS = re.compile(
    r"AIza[0-9A-Za-z_\-]{35}"
    r"|sk-ant-[0-9A-Za-z_\-]{20,}"
    r"|sk-[0-9A-Za-z]{20,}"
    r"|(api_key|secret|token|password)\s*=\s*[\"'][^\"']{16,}[\"']",
    re.IGNORECASE,
)
ASSERTS = re.compile(r"^\s*(assert\b|pytest\.raises|pytest\.approx)")
EXCEPTION_ROW = re.compile(r"^\|\s*E\d+\s*\|")

# Luật chỉ áp cho code Python; tài liệu hay nhắc tới chính các mẫu này.
CODE_RULES = [
    ("silenced-checker", SUPPRESSIONS),
    ("test-made-easier", SKIPS),
    ("unfinished-work", STUBS),
]


@dataclass(frozen=True)
class Line:
    file: str
    text: str


@dataclass(frozen=True)
class Violation:
    rule: str
    file: str
    text: str


def parse_diff(diff: str) -> tuple[list[Line], list[Line]]:
    """Tách diff unified thành các dòng thêm và dòng xoá, kèm tên tệp."""
    added, removed = [], []
    new_file = old_file = ""
    in_hunk = False  # "--- "/"+++ " chỉ là tên tệp khi ở phần đầu, trước hunk đầu tiên
    for raw in diff.splitlines():
        if raw.startswith("diff "):
            in_hunk = False
        elif raw.startswith("@@"):
            in_hunk = True
        elif not in_hunk and raw.startswith("+++ "):
            new_file = raw[4:].removeprefix("b/")
        elif not in_hunk and raw.startswith("--- "):
            old_file = raw[4:].removeprefix("a/")
        elif in_hunk and raw.startswith("+"):
            added.append(Line(new_file, raw[1:]))
        elif in_hunk and raw.startswith("-"):
            removed.append(Line(old_file, raw[1:]))
    return added, removed


def _is_test(path: str) -> bool:
    return path.startswith("tests/") or path.rsplit("/", 1)[-1].startswith("test_")


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)]


def _cells(text: str) -> list[str]:
    return [c.strip() for c in text.strip().strip("|").split("|")]


def _loosened(old: str, new: str) -> bool:
    """Dòng bảng bị nới theo hướng ghi ở ô cuối. Dòng không ghi hướng thì không xét."""
    direction = _cells(new)[-1]
    pairs = list(zip(_numbers(new), _numbers(old), strict=False))
    if "không được giảm" in direction:
        return any(n < o for n, o in pairs)
    if "không được tăng" in direction:
        return any(n > o for n, o in pairs)
    return False


def _assert_count(lines: list[Line]) -> Counter:
    return Counter(line.file for line in lines if _is_test(line.file) and ASSERTS.match(line.text))


def _line_violations(added: list[Line]) -> list[Violation]:
    found = []
    for line in added:
        if line.file in EXEMPT:
            continue
        short = line.text.strip()[:120]
        if line.file.rsplit("/", 1)[-1] == ".env" or SECRETS.search(line.text):
            found.append(Violation("secret", line.file, "[đã ẩn giá trị]"))
        elif line.file.endswith(".py"):
            found += [
                Violation(rule, line.file, short) for rule, pattern in CODE_RULES if pattern.search(line.text)
            ]
        elif line.file == "CONSTRAINTS.md" and EXCEPTION_ROW.match(line.text):
            found.append(Violation("new-exception", line.file, short))
    return found


def _lost_assertions(added: list[Line], removed: list[Line]) -> list[Violation]:
    """Sửa một assert là xoá 1 dòng và thêm 1 dòng; chỉ báo khi số assert trong tệp giảm thật."""
    kept = _assert_count(added)
    return [
        Violation("assertion-removed", file, f"mất {lost - kept[file]} assert")
        for file, lost in _assert_count(removed).items()
        if lost > kept[file]
    ]


def _table_rows(lines: list[Line]) -> list[str]:
    return [line.text for line in lines if line.file == "CONSTRAINTS.md" and line.text.startswith("|")]


def _loosened_rows(added: list[Line], removed: list[Line]) -> list[Violation]:
    new_rows = {_cells(text)[0]: text for text in _table_rows(added)}
    found = []
    for old in _table_rows(removed):
        new = new_rows.get(_cells(old)[0])
        if new and _loosened(old, new):
            found.append(
                Violation("threshold-lowered", "CONSTRAINTS.md", f"{old.strip()}  ->  {new.strip()}")
            )
    return found


def find_violations(added: list[Line], removed: list[Line]) -> list[Violation]:
    return _line_violations(added) + _lost_assertions(added, removed) + _loosened_rows(added, removed)
