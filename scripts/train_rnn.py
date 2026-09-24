"""LSTM so với TF-IDF + hồi quy logistic cho 2 bài toán: nhóm ngành và khoảng lương.

Dùng tập chia của Task 2. LSTM chọn epoch, hồi quy logistic chọn C, đều theo macro-F1 trên val.
Kết quả trên test ghi vào eval/results/rnn.json.

python scripts/train_rnn.py
"""

import json
import subprocess
import time
from datetime import date

import numpy as np
import pandas as pd
import torch

from career_advisor.config import EVAL_DIR, PROCESSED_DIR, SEED
from career_advisor.rnn.train import classification_report, posting_texts, tfidf_logreg, train_lstm

SALARY_BANDS = ["under_7m", "7_10m", "10_15m", "15_20m", "20_30m", "30_50m", "50m_plus"]
MAX_EPOCHS = 10


def run_task(
    name: str, texts: pd.Series, target: pd.Series, split: pd.Series, labels: list[str], device
) -> dict:
    keep = target.notna()
    y = target[keep].map({label: i for i, label in enumerate(labels)}).astype(int)
    texts, split = texts[keep], split[keep]
    part = {s: split == s for s in ("train", "val", "test")}
    print(f"\n== {name}: " + ", ".join(f"{s} {int(m.sum())}" for s, m in part.items()), flush=True)
    majority = np.full(int(part["test"].sum()), y[part["train"]].mode()[0])
    result = {
        "n": {s: int(m.sum()) for s, m in part.items()},
        "majority": classification_report(y[part["test"]], majority, labels),
    }

    start = time.perf_counter()
    preds, c = tfidf_logreg(
        texts[part["train"]], y[part["train"]], texts[part["val"]], y[part["val"]], texts[part["test"]]
    )
    result["tfidf_logreg"] = {
        **classification_report(y[part["test"]], preds, labels),
        "C": c,
        "seconds": round(time.perf_counter() - start, 1),
    }
    lr = result["tfidf_logreg"]
    print(f"TF-IDF + LR: acc {lr['accuracy']:.3f}, F1 {lr['macro_f1']:.3f}, C {c}", flush=True)

    start = time.perf_counter()
    lstm = train_lstm(
        texts[part["train"]], y[part["train"]].to_numpy(), texts[part["val"]], y[part["val"]].to_numpy(),
        n_classes=len(labels), epochs=MAX_EPOCHS, device=device, seed=SEED,
        log=lambda h: print(" ", h, flush=True),
    )  # fmt: skip
    preds = lstm.predict(texts[part["test"]])
    result["lstm"] = {
        **classification_report(y[part["test"]], preds, labels),
        "best_epoch": lstm.best_epoch,
        "history": lstm.history,
        "max_len": lstm.max_len,
        "vocab_size": len(lstm.vocab),
        "parameters": sum(p.numel() for p in lstm.model.parameters()),
        "seconds": round(time.perf_counter() - start, 1),
    }
    print(f"LSTM: acc {result['lstm']['accuracy']:.3f}, F1 {result['lstm']['macro_f1']:.3f}", flush=True)
    return result


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    split = pd.read_parquet(PROCESSED_DIR / "splits.parquet")["split"].reindex(postings.index).astype(str)
    texts = posting_texts(postings)
    categories = sorted(postings["category"].unique())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    result = {
        "date": str(date.today()),
        "commit": commit.strip(),
        "seed": SEED,
        "device": device,
        "input": "job_title_norm + description + requirements_text",
        "category": run_task("category", texts, postings["category"], split, categories, device),
        "salary_band": run_task("salary_band", texts, postings["salary_band"], split, SALARY_BANDS, device),
    }
    (EVAL_DIR / "results" / "rnn.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
