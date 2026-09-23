"""So sánh các LLM chạy cục bộ trên việc chấm cặp kỹ năng (phần dev của bộ nhãn tay).

    python scripts/benchmark_llm.py qwen3.5:9b gemma4:12b

Đo: thời gian mỗi lượt gọi, precision/recall so với nhãn tay, tỷ lệ lô JSON hỏng hoặc thiếu,
và RAM model chiếm trong Ollama. Kết quả gộp vào eval/results/llm_benchmark.json.
"""

import json
import sys
import time

from career_advisor.cleaning.resolve import JUDGE_SYSTEM, LLM_BATCH, judge_prompt, read_judgements
from career_advisor.config import CACHE_DIR, EVAL_DIR
from career_advisor.evaluation.entity_resolution import labeled_pairs
from career_advisor.evaluation.metrics import precision_recall
from career_advisor.evaluation.skill_pairs import load_sheet
from career_advisor.llm import DEFAULT_OLLAMA_URL, OllamaClient

OUT = EVAL_DIR / "results" / "llm_benchmark.json"


def ram_gb(model: str) -> float | None:
    import urllib.request

    with urllib.request.urlopen(f"{DEFAULT_OLLAMA_URL}/api/ps") as resp:
        running = json.loads(resp.read())["models"]
    sizes = [m["size"] for m in running if m["name"].startswith(model)]
    return round(sizes[0] / 1e9, 2) if sizes else None


def benchmark(model: str, pairs) -> dict:
    client = OllamaClient(model, cache_dir=CACHE_DIR / "llm")
    batches = [pairs.iloc[i : i + LLM_BATCH] for i in range(0, len(pairs), LLM_BATCH)]
    client.complete("Trả lời 1 từ: 2+2=?")  # nạp model vào RAM trước khi bấm giờ
    verdicts, seconds, broken = [], [], 0
    for batch in batches:
        rows = list(zip(batch["a"], batch["b"], strict=True))
        start = time.time()
        text = client.complete(judge_prompt(rows), system=JUDGE_SYSTEM, json_output=True)
        seconds.append(time.time() - start)
        result, complete = read_judgements(text, len(rows))
        verdicts.extend(result)
        broken += not complete
    metrics = precision_recall(pairs["truth"].tolist(), verdicts)
    return {
        "model": model,
        "n_pairs": len(pairs),
        "n_calls": len(batches),
        "seconds_per_call": round(sum(seconds) / len(seconds), 1),
        "broken_json_rate": round(broken / len(batches), 3),
        "ram_gb": ram_gb(model),
        **metrics,
    }


if __name__ == "__main__":
    pairs = labeled_pairs(load_sheet(EVAL_DIR / "labels" / "skill_pairs.csv"))
    dev = pairs[pairs["split"] == "dev"].reset_index(drop=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model in sys.argv[1:]:
        results[model] = benchmark(model, dev)
        print(json.dumps(results[model], ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
