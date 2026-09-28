"""Gom `eval/results/*.json` thành `docs/results.md`.

.venv/bin/python scripts/summarize_results.py
"""

import json

from career_advisor.config import EVAL_DIR, ROOT
from career_advisor.evaluation.summary import SECTIONS, build_report

OUT = ROOT / "docs" / "results.md"


def main() -> None:
    results = {}
    for _, name, _ in SECTIONS:
        path = EVAL_DIR / "results" / f"{name}.json"
        if path.exists():
            # pandas ghi NaN cho ô trống (ví dụ trích dẫn của câu ngoài phạm vi); JSON chuẩn không có NaN.
            results[name] = json.loads(path.read_text(encoding="utf-8").replace("NaN", "null"))
    OUT.write_text(build_report(results), encoding="utf-8")
    print(f"Đã ghi {OUT.relative_to(ROOT)} ({len(results)}/{len(SECTIONS)} phần có dữ liệu).")


if __name__ == "__main__":
    main()
