"""Gán nhãn tay cho cặp kỹ năng: cùng một kỹ năng hay khác nhau.

    python scripts/label_skill_pairs.py make   # tạo tệp nhãn (chỉ chạy 1 lần)
    python scripts/label_skill_pairs.py        # gán nhãn, tắt đi mở lại thì làm tiếp

Phím: y = cùng kỹ năng, n = khác, s = bỏ qua (không chắc), u = sửa cặp vừa gán, q = thoát.
"""

import sys

import numpy as np
import pandas as pd

from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.evaluation.cli import read_key
from career_advisor.evaluation.skill_pairs import (
    build_label_sheet,
    load_sheet,
    next_unlabeled,
    sample_cosine_pairs,
    sample_rule_pairs,
    save_sheet,
    set_label,
)

SHEET = EVAL_DIR / "labels" / "skill_pairs.csv"
KEYS = {"y": "same", "n": "different", "s": "skip"}
GUIDE = """
QUY TẮC: chọn "cùng" (y) khi hai tên thay được cho nhau trong yêu cầu tuyển dụng —
người có kỹ năng A đáp ứng tin yêu cầu B, và ngược lại.
  y: "reactjs" – "react", "bảo dưỡng máy móc" – "bảo trì máy móc", "hsk5" – "hsk 5"
  n: "java" – "javascript", "tiếng anh" – "tiếng trung", "chỉnh sửa ảnh video" – "chỉnh sửa video"
     (một bên rộng hơn hẳn bên kia cũng chọn n)
  s: không đủ hiểu để quyết
"""


def make() -> None:
    if SHEET.exists() and (load_sheet(SHEET)["label"] != "").any():
        sys.exit(f"{SHEET} đã có nhãn. Không tạo lại để khỏi mất nhãn.")
    skills = pd.read_parquet(PROCESSED_DIR / "emb" / "skills_index.parquet")
    chunks = sorted((PROCESSED_DIR / "emb" / "skills").glob("0*.npy"))
    vectors = np.concatenate([np.load(c) for c in chunks])
    skill_map = pd.read_parquet(PROCESSED_DIR / "skill_map.parquet")
    pairs = pd.concat([sample_cosine_pairs(skills, vectors), sample_rule_pairs(skill_map)], ignore_index=True)
    sheet = build_label_sheet(pairs)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    save_sheet(sheet, SHEET)
    print(f"Đã tạo {len(sheet)} cặp:", sheet["stratum"].value_counts().sort_index().to_dict())


def context() -> tuple[dict[str, int], dict[str, list[str]]]:
    skills = pd.read_parquet(PROCESSED_DIR / "emb" / "skills_index.parquet")
    skill_map = pd.read_parquet(PROCESSED_DIR / "skill_map.parquet")
    forms = skill_map.groupby("canonical")["raw"].apply(lambda r: r.head(3).tolist()).to_dict()
    return dict(zip(skills["skill"], skills["count"], strict=True)), forms


def label() -> None:
    sheet = load_sheet(SHEET)
    counts, forms = context()
    print(GUIDE)
    history: list[str] = []
    while (pos := next_unlabeled(sheet)) is not None:
        row = sheet.iloc[pos]
        done = int((sheet["label"] != "").sum())
        print(f"\n[{done + 1}/{len(sheet)}]")
        for side in ("a", "b"):
            name = row[side]
            extra = f"  ({counts[name]} tin; ví dụ: {forms.get(name, [])})" if name in counts else ""
            print(f"  {side.upper()}: {name}{extra}")
        print("  cùng (y) / khác (n) / bỏ qua (s) / sửa cặp trước (u) / thoát (q)? ", end="", flush=True)
        key = read_key()
        print(key)
        if key == "q":
            break
        if key == "u" and history:
            sheet.loc[sheet["pair_id"] == history.pop(), "label"] = ""
        elif key in KEYS:
            sheet = set_label(sheet, row["pair_id"], KEYS[key])
            history.append(row["pair_id"])
        else:
            continue
        save_sheet(sheet, SHEET)
    done = int((sheet["label"] != "").sum())
    print(f"\nĐã gán {done}/{len(sheet)} cặp. Nhãn lưu ở {SHEET}.")


if __name__ == "__main__":
    make() if sys.argv[1:] == ["make"] else label()
