"""Kiểm tay 50 tin: các trường đã trích (kỹ năng, lương, kinh nghiệm, bằng cấp) có đúng với tin gốc không.

    python scripts/label_data_quality.py make   # tạo tệp nhãn cho 50 tin đầu của mẫu 200 tin
    python scripts/label_data_quality.py        # kiểm, tắt đi mở lại thì làm tiếp

Mỗi trường: y = đúng, n = sai, k = không rõ (tin gốc không đủ để kiểm).
Phím khác: u = làm lại tin trước, q = thoát. Nhãn lưu sau mỗi tin.
"""

import sys

import pandas as pd

from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.data import load_postings
from career_advisor.evaluation.cli import read_key
from career_advisor.evaluation.data_quality import (
    FIELD_NAMES,
    FIELDS,
    RULES,
    extracted_fields,
    sample_ids,
    source_text,
)

HUMAN = EVAL_DIR / "labels" / "data_quality.csv"
N_SAMPLE, N_HUMAN = 200, 50
KEYS = {"y": "correct", "n": "wrong", "k": "unclear"}


def load() -> pd.DataFrame:
    return pd.read_csv(HUMAN, dtype=str, keep_default_na=False)


def save(sheet: pd.DataFrame) -> None:
    tmp = HUMAN.with_suffix(".tmp")
    sheet.to_csv(tmp, index=False)
    tmp.replace(HUMAN)


def make() -> None:
    if HUMAN.exists() and (load()[list(FIELDS)] != "").any().any():
        sys.exit(f"{HUMAN} đã có nhãn. Không tạo lại để khỏi mất nhãn.")
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    ids = sample_ids(postings.index, N_SAMPLE)[:N_HUMAN]
    save(pd.DataFrame({"posting_id": ids, **{f: "" for f in FIELDS}}))
    print(f"Đã tạo {N_HUMAN} tin ở {HUMAN}")


def label() -> None:
    sheet = load()
    raw, postings = load_postings(), pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    skills = pd.read_parquet(PROCESSED_DIR / "stats" / "posting_skills.parquet")
    print("\nQUY TẮC (giống hệt quy tắc đưa cho LLM):\n" + RULES.rsplit("\nTrả về JSON", 1)[0])
    history: list[int] = []
    while (todo := sheet.index[(sheet[list(FIELDS)] == "").any(axis=1)]).size:
        i = int(todo[0])
        pid = int(sheet.loc[i, "posting_id"])
        fields = extracted_fields(postings.loc[pid], skills[skills["posting_id"] == pid])
        done = len(sheet) - len(todo)
        print(f"\n{'=' * 80}\n[{done + 1}/{len(sheet)}] TIN #{pid}\n{source_text(raw.loc[pid])}\n{'-' * 80}")
        answers, quit_now, undo = {}, False, False
        for f in FIELDS:
            print(
                f"  {FIELD_NAMES[f]}: {fields[f]}\n    đúng (y) / sai (n) / không rõ (k) / u / q? ",
                end="",
                flush=True,
            )
            while (key := read_key()) not in (*KEYS, "u", "q"):
                pass
            print(key)
            if key == "q":
                quit_now = True
                break
            if key == "u":
                undo = True
                break
            answers[f] = KEYS[key]
        if quit_now:
            break
        if undo:
            if history:
                sheet.loc[history.pop(), list(FIELDS)] = ""
                save(sheet)
            continue
        for f, verdict in answers.items():
            sheet.loc[i, f] = verdict
        history.append(i)
        save(sheet)
    done = int((sheet[list(FIELDS)] != "").all(axis=1).sum())
    print(f"\nĐã kiểm {done}/{len(sheet)} tin. Nhãn lưu ở {HUMAN}.")


if __name__ == "__main__":
    make() if sys.argv[1:] == ["make"] else label()
