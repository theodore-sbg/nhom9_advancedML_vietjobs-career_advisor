"""Đánh giá từng module. Kết quả ghi vào eval/results/.

python scripts/evaluate.py er-thresholds       # chọn ngưỡng gộp tên trên phần dev
python scripts/evaluate.py entity-resolution   # 4 hệ gộp tên, chỉ số trên dev và test
"""

import json
import subprocess
import sys
from datetime import date

import pandas as pd

from career_advisor.cleaning.resolve import AUTO_THRESHOLD, LLM_KINDS, LLM_LOW, judge_pairs
from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.evaluation.entity_resolution import choose_threshold, labeled_pairs, predict_same
from career_advisor.evaluation.metrics import precision_recall
from career_advisor.evaluation.skill_pairs import load_sheet
from career_advisor.llm import make_client

RESULTS = EVAL_DIR / "results"
AUTO_CANDIDATES = (0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95)
LOW_CANDIDATES = (0.78, 0.80, 0.82, 0.84, 0.86, 0.88)


def _meta(client) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"date": str(date.today()), "commit": commit, "llm": f"{client.provider}:{client.model}"}


def _judged_pairs():
    client = make_client()
    pairs = labeled_pairs(load_sheet(EVAL_DIR / "labels" / "skill_pairs.csv"))
    pairs["llm_same"] = judge_pairs(client, list(zip(pairs["a"], pairs["b"], strict=True)))
    return client, pairs


def er_thresholds() -> None:
    client, pairs = _judged_pairs()
    dev = pairs[(pairs["split"] == "dev") & (pairs["stratum"] != "rule")].reset_index(drop=True)

    auto, auto_table = choose_threshold(dev["truth"], lambda t: dev["cosine"] >= t, AUTO_CANDIDATES)
    print("Tự gộp khi cosine ≥ t (dev):\n", auto_table.round(3).to_string(index=False))
    upper = auto if auto is not None else max(AUTO_CANDIDATES)
    kinds = pd.read_parquet(PROCESSED_DIR / "emb" / "skills_index.parquet").set_index("skill")["kind"]
    dev["llm_kind"] = dev["a"].map(kinds).isin(LLM_KINDS) & dev["b"].map(kinds).isin(LLM_KINDS)
    for label, rows in (("mọi cặp", dev), (f"chỉ loại {LLM_KINDS}", dev[dev["llm_kind"]])):
        band = rows[rows["cosine"] < upper].reset_index(drop=True)
        low, low_table = choose_threshold(
            band["truth"], lambda t, b=band: (b["cosine"] >= t) & b["llm_same"], LOW_CANDIDATES
        )
        print(f"LLM trong vùng [t, {upper}), {label} (dev):\n", low_table.round(3).to_string(index=False))
    result = {
        **_meta(client),
        "auto_threshold": auto,
        "llm_low": low,
        "n_dev": len(dev),
        "auto_table": auto_table.to_dict("records"),
        "llm_table": low_table.to_dict("records"),
    }
    (RESULTS / "er_thresholds.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Chọn: AUTO_THRESHOLD = {auto}, LLM_LOW = {low}. Lượt gọi thật: {client.calls}")


def entity_resolution() -> None:
    client, pairs = _judged_pairs()
    systems = {
        "rule": predict_same(pairs, pd.DataFrame(columns=["skill", "canonical"])),
        "rule+embedding": predict_same(
            pairs, pd.read_parquet(PROCESSED_DIR / "skill_merges_embedding.parquet")
        ),
        "llm_judge_only": pairs["llm_same"],
        "rule+embedding+llm": predict_same(pairs, pd.read_parquet(PROCESSED_DIR / "skill_merges.parquet")),
    }
    rows = []
    for split in ("dev", "test"):
        mask = (pairs["split"] == split).to_numpy()
        for name, pred in systems.items():
            metrics = precision_recall(pairs["truth"][mask].tolist(), pred[mask].tolist())
            rows.append({"split": split, "system": name, **metrics})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    result = {**_meta(client), "auto_threshold": AUTO_THRESHOLD, "llm_low": LLM_LOW, "rows": rows}
    (RESULTS / "entity_resolution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"er-thresholds": er_thresholds, "entity-resolution": entity_resolution}[sys.argv[1]]()
