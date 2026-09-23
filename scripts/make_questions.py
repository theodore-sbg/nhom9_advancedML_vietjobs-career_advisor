"""Sinh bộ câu hỏi kiểm thử → eval/questions/questions.jsonl. Đáp án tính bằng pandas."""

import json
import time

import pandas as pd

from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.evaluation.questions import build_question_set

OUT = EVAL_DIR / "questions" / "questions.jsonl"

if __name__ == "__main__":
    start = time.time()
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    skills = pd.read_parquet(PROCESSED_DIR / "stats" / "posting_skills.parquet")[["posting_id", "skill"]]
    questions = build_question_set(postings, skills)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    frame = pd.DataFrame(questions)
    print(f"{len(questions)} câu trong {time.time() - start:.0f} giây → {OUT}")
    print(frame.groupby(["group", "split"]).size().unstack().to_string())
