"""Gán nhãn tay độ phù hợp CV–tin (0/1/2) cho khoảng 100 cặp, để đo độ khớp giữa LLM và người.

    python scripts/label_relevance.py make   # lấy mẫu sau khi đã chạy `evaluate.py retrieval`
    python scripts/label_relevance.py        # gán nhãn, tắt đi mở lại thì làm tiếp

Phím: 2 = phù hợp, 1 = phù hợp một phần, 0 = không phù hợp, s = bỏ qua, u = sửa cặp trước, q = thoát.
"""

import json
import sys

import pandas as pd

from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.evaluation.cli import read_key
from career_advisor.evaluation.relevance import MAX_REQUIREMENTS_CHARS, sample_for_human

LLM = EVAL_DIR / "labels" / "relevance_llm.csv"
HUMAN = EVAL_DIR / "labels" / "relevance_human.csv"
GUIDE = """
QUY TẮC (giống hệt quy tắc đưa cho LLM):
  2: đúng nghề hoặc lĩnh vực của ứng viên, và CV đáp ứng phần lớn yêu cầu chính
  1: cùng lĩnh vực nhưng lệch vai trò hoặc cấp độ, hoặc thiếu nhiều kỹ năng chính
  0: không phù hợp
  Không xét nơi làm, giới tính, tuổi hay mức lương: chỉ xét nghề nghiệp và kỹ năng.
"""


def load() -> pd.DataFrame:
    return pd.read_csv(HUMAN, dtype={"label": str}, keep_default_na=False)


def save(sheet: pd.DataFrame) -> None:
    tmp = HUMAN.with_suffix(".tmp")
    sheet.to_csv(tmp, index=False)
    tmp.replace(HUMAN)


def make() -> None:
    if HUMAN.exists() and (load()["label"] != "").any():
        sys.exit(f"{HUMAN} đã có nhãn. Không tạo lại để khỏi mất nhãn.")
    save(sample_for_human(pd.read_csv(LLM)))
    print(f"Đã tạo {len(load())} cặp ở {HUMAN}")


def label() -> None:
    sheet = load()
    cvs = {
        c["id"]: c["text"] for c in map(json.loads, (EVAL_DIR / "cvs" / "cvs.jsonl").read_text().splitlines())
    }
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    print(GUIDE)
    history: list[int] = []
    while (empty := sheet.index[sheet["label"] == ""]).size:
        i = int(empty[0])
        row, post = sheet.loc[i], postings.loc[int(sheet.loc[i, "posting_id"])]
        print(f"\n[{int((sheet['label'] != '').sum()) + 1}/{len(sheet)}]  CV: {cvs[row['cv_id']]}")
        print(f"  TIN #{row['posting_id']}: {post['job_title']}")
        print(f"  Yêu cầu: {str(post['requirements_text'])[:MAX_REQUIREMENTS_CHARS]}")
        print("  2 / 1 / 0 / bỏ qua (s) / sửa cặp trước (u) / thoát (q)? ", end="", flush=True)
        key = read_key()
        print(key)
        if key == "q":
            break
        if key == "u" and history:
            sheet.loc[history.pop(), "label"] = ""
        elif key in ("0", "1", "2", "s"):
            sheet.loc[i, "label"] = "skip" if key == "s" else key
            history.append(i)
        else:
            continue
        save(sheet)
    print(f"\nĐã gán {int((sheet['label'] != '').sum())}/{len(sheet)} cặp. Nhãn lưu ở {HUMAN}.")


if __name__ == "__main__":
    make() if sys.argv[1:] == ["make"] else label()
