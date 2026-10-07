"""Kiểm phần nền của CONSTRAINTS.md trên thay đổi so với một mốc git.

    .venv/bin/python scripts/floor_guard.py              # so với HEAD: mọi thay đổi chưa commit
    .venv/bin/python scripts/floor_guard.py --base main~3

Mã thoát: 0 sạch, 1 có vi phạm (chặn), 2 không chạy được (không phải repo git, mốc sai).
"""

import argparse
import subprocess
import sys

from career_advisor.floor_guard import find_violations, parse_diff


def _git(*args: str, check: bool = True) -> str:
    # quotePath=false: tên tệp tiếng Việt không bị đổi thành "th\\341..." trong ngoặc kép.
    result = subprocess.run(["git", "-c", "core.quotePath=false", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def collect_diff(base: str) -> str:
    """Diff so với mốc, cộng cả tệp mới chưa được git theo dõi."""
    parts = [_git("diff", "--unified=0", base, "--")]
    # -z: tách tên bằng \\0, không bị ngoặc kép hay ký tự đặc biệt làm hỏng tên.
    for path in filter(None, _git("ls-files", "-z", "--others", "--exclude-standard").split("\0")):
        parts.append(_git("diff", "--no-index", "--unified=0", "/dev/null", path, check=False))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="HEAD")
    args = parser.parse_args()
    try:
        _git("rev-parse", "--verify", args.base)
        diff = collect_diff(args.base)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"floor-guard: không chạy được ({exc})", file=sys.stderr)
        return 2

    violations = find_violations(*parse_diff(diff))
    if not violations:
        print("floor-guard: sạch")
        return 0
    print(f"floor-guard: {len(violations)} vi phạm phần nền:", file=sys.stderr)
    for v in violations:
        print(f"  [{v.rule}] {v.file}: {v.text}", file=sys.stderr)
    print(
        "\nSửa code, hoặc ghi một ngoại lệ có người chịu trách nhiệm trong CONSTRAINTS.md.", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
