"""Đánh giá từng module. Kết quả ghi vào eval/results/.

python scripts/evaluate.py er-thresholds       # chọn ngưỡng gộp tên trên phần dev
python scripts/evaluate.py entity-resolution   # 4 hệ gộp tên, chỉ số trên dev và test
python scripts/evaluate.py retrieval           # 5 cách truy xuất CV → tin, LLM chấm độ phù hợp
python scripts/evaluate.py kappa               # độ khớp giữa điểm LLM và nhãn tay (người, Claude)
python scripts/evaluate.py kg-rag              # KG-RAG so với vector RAG trên bộ 50 câu hỏi
python scripts/evaluate.py ablation            # 1 agent so với 4 agent, có và không có gộp tên kỹ năng
python scripts/evaluate.py latency             # thời gian trả lời của luồng 4 agent, không dùng cache
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


RELEVANCE_LLM = EVAL_DIR / "labels" / "relevance_llm.csv"
RELEVANCE_HUMAN = EVAL_DIR / "labels" / "relevance_human.csv"
# Nhãn do Claude gán khi người dùng chưa gán kịp. Báo riêng, không bao giờ gộp vào nhãn người.
RELEVANCE_CLAUDE = EVAL_DIR / "labels" / "relevance_claude.csv"
ANNOTATORS = {"human": RELEVANCE_HUMAN, "claude": RELEVANCE_CLAUDE}


def _agreement(llm: pd.DataFrame) -> dict:
    """κ giữa điểm LLM và từng bộ nhãn tay (người, Claude), kèm ma trận nhầm lẫn."""
    from career_advisor.evaluation.metrics import cohen_kappa

    out = {}
    for name, path in ANNOTATORS.items():
        if not path.exists():
            continue
        labels = pd.read_csv(path, dtype={"label": str}, keep_default_na=False)
        labels = labels[labels["label"].isin(["0", "1", "2"])]
        both = labels.merge(llm.dropna(subset=["grade"]), on=["cv_id", "posting_id"])
        if both.empty:
            continue
        a, b = both["label"].astype(int).tolist(), both["grade"].astype(int).tolist()
        confusion = pd.crosstab(pd.Series(a, name=name), pd.Series(b, name="llm"))
        out[f"{name}_vs_llm"] = {
            "n": len(both),
            "unweighted": cohen_kappa(a, b),
            "linear": cohen_kappa(a, b, "linear"),
            "exact_agreement": float((pd.Series(a) == pd.Series(b)).mean()),
            "confusion": {
                str(k): {str(c): int(v) for c, v in row.items()} for k, row in confusion.iterrows()
            },
        }
    return out


def kappa() -> None:
    result = _agreement(pd.read_csv(RELEVANCE_LLM))
    for name, stats in result.items():
        print(
            f"{name}: n={stats['n']}, κ={stats['unweighted']:.3f}, κ tuyến tính={stats['linear']:.3f}, "
            f"trùng khớp={stats['exact_agreement']:.1%}"
        )
        print("  ma trận (hàng: nhãn tay, cột: LLM):", stats["confusion"])
    path = RESULTS / "retrieval.json"
    if path.exists():
        saved = json.loads(path.read_text())
        saved.pop("kappa", None)
        saved["agreement"] = result
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))


def retrieval() -> None:
    from career_advisor.evaluation.relevance import judge_relevance, posting_summary
    from career_advisor.evaluation.retrieval_eval import K, pool, score_systems
    from career_advisor.retrieval.factory import build_searchers, load_resources

    cvs = [
        json.loads(line) for line in (EVAL_DIR / "cvs" / "cvs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resources = load_resources()
    searchers = build_searchers(resources)
    rankings = {
        cv["id"]: {
            name: s.search(cv["text"], k=K)["posting_id"].astype(int).tolist()
            for name, s in searchers.items()
        }
        for cv in cvs
    }
    (RESULTS / "retrieval_rankings.json").write_text(json.dumps(rankings, indent=1))

    postings, skills = resources["postings"], resources["skills"]
    skills_of = skills.groupby("posting_id")["skill"].apply(list)
    pools = pool(rankings)
    client = make_client()
    total = sum(len(v) for v in pools.values())
    print(f"Chấm {total} cặp CV–tin bằng {client.model}", flush=True)
    rows, grades = [], {}
    for cv in cvs:
        items = [(pid, posting_summary(postings.loc[pid], skills_of.get(pid, []))) for pid in pools[cv["id"]]]
        grades[cv["id"]] = judge_relevance(client, cv["text"], items)
        rows += [(cv["id"], pid, g) for pid, g in grades[cv["id"]].items()]
        print(f"  {cv['id']}: {len(items)} tin, lượt gọi thật {client.calls}", flush=True)
    pd.DataFrame(rows, columns=["cv_id", "posting_id", "grade"]).to_csv(RELEVANCE_LLM, index=False)

    split_of = {cv["id"]: cv["split"] for cv in cvs}
    tables = []
    for split in ("dev", "test"):
        subset = {cv: r for cv, r in rankings.items() if split_of[cv] == split}
        tables.append(score_systems(subset, grades).assign(split=split))
    table = pd.concat(tables, ignore_index=True)
    print(table.round(3).to_string(index=False))

    result = {**_meta(client), "k": K, "n_pairs_judged": total, "rows": table.to_dict("records")}
    result["agreement"] = _agreement(pd.DataFrame(rows, columns=["cv_id", "posting_id", "grade"]))
    (RESULTS / "retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def load_questions() -> list[dict]:
    path = EVAL_DIR / "questions" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_qa(systems: dict, out_name: str) -> None:
    """Chạy mỗi hệ (tên → hàm câu hỏi → Answer) trên bộ câu hỏi, chấm và ghi kết quả."""
    from career_advisor.evaluation.qa_eval import aggregate, score_answer

    questions = load_questions()
    rows, answers = [], []
    for q in questions:
        for name, ask in systems.items():
            ans = ask(q["question"])
            record = {
                "text": ans.text,
                "refuse": ans.refuse,
                "citations": ans.citations,
                "allowed_ids": ans.allowed_ids,
            }
            score = score_answer(q, record)
            rows.append({"system": name, "group": q["group"], "split": q["split"], **score})
            answers.append({"id": q["id"], "system": name, "question": q["question"], **record, **score})
        print(f"  {q['id']} xong", flush=True)
    (RESULTS / f"{out_name}_answers.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8"
    )
    tables = []
    for split in ("dev", "test"):
        tables.append(aggregate([r for r in rows if r["split"] == split]).assign(split=split))
    table = pd.concat(tables, ignore_index=True)
    print(table.round(3).to_string(index=False))
    client = make_client()
    result = {**_meta(client), "rows": table.to_dict("records")}
    (RESULTS / f"{out_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def kg_rag() -> None:
    import pickle

    import numpy as np

    from career_advisor.embeddings import BgeM3Encoder
    from career_advisor.rag.answer import answer_question
    from career_advisor.rag.vector_rag import vector_answer
    from career_advisor.retrieval.dense import DenseSearcher

    with (PROCESSED_DIR / "graph.pkl").open("rb") as f:
        graph = pickle.load(f)
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    skills = pd.read_parquet(PROCESSED_DIR / "stats" / "posting_skills.parquet")
    skills_of = skills.groupby("posting_id")["skill"].apply(list).to_dict()
    dense = DenseSearcher(
        np.load(PROCESSED_DIR / "emb" / "postings.npy"), postings.index.to_numpy(), BgeM3Encoder()
    )
    client = make_client()
    run_qa(
        {
            "kg_rag": lambda question: answer_question(client, graph, question),
            "vector_rag": lambda question: vector_answer(client, dense, postings, skills_of, question),
        },
        "qa_kg_vs_vector",
    )


def _load_graph(name: str):
    import pickle

    with (PROCESSED_DIR / name).open("rb") as f:
        return pickle.load(f)


def _agent_record(result) -> dict:
    return {
        "verified": result.verification.ok,
        "attempts": result.attempts,
        "fallback": result.fallback,
        "llm_calls": result.llm_calls,
        "seconds": round(result.seconds, 2),
    }


def _qa_ablation(agents: dict) -> tuple[pd.DataFrame, list[dict]]:
    from career_advisor.evaluation.qa_eval import aggregate, score_answer

    rows = []
    for q in load_questions():
        for name, agent in agents.items():
            result = agent.run(question=q["question"])
            ans = result.answer
            record = {"text": ans.text, "refuse": ans.refuse, "citations": ans.citations}
            record["allowed_ids"] = ans.allowed_ids
            rows.append(
                {
                    "part": "qa",
                    "id": q["id"],
                    "system": name,
                    "group": q["group"],
                    "split": q["split"],
                    **record,
                    **score_answer(q, record),
                    **_agent_record(result),
                }
            )
        print(f"  {q['id']} xong", flush=True)
    frame = pd.DataFrame(rows)
    tables = []
    for (split, _, _), g in frame.groupby(["split", "system", "group"]):
        base = aggregate(g.to_dict("records")).iloc[0].to_dict()
        base |= {
            "split": split,
            "verified_rate": float(g["verified"].mean()),
            "fallback_rate": float(g["fallback"].mean()),
            "mean_llm_calls": float(g["llm_calls"].mean()),
        }
        tables.append(base)
    return pd.DataFrame(tables), rows


def _cv_ablation(agents: dict, G) -> tuple[pd.DataFrame, list[dict]]:
    from career_advisor.evaluation.agent_eval import dominant_category, mentioned_share
    from career_advisor.graph import query as gq
    from career_advisor.rag.subgraph import group_node

    # Chỉ CV test: CV dev đã dùng để chỉnh prompt lộ trình.
    cvs = [json.loads(line) for line in (EVAL_DIR / "cvs" / "cvs.jsonl").read_text("utf-8").splitlines()]
    cvs = [cv for cv in cvs if cv["split"] == "test"]
    rows = []
    for cv in cvs:
        for name, agent in agents.items():
            result = agent.run(cv_text=cv["text"])
            ans = result.answer
            group = group_node(ans.linked)
            missing = [s.skill for s in gq.missing_skills(G, ans.linked.skills, group, k=3)] if group else []
            rows.append(
                {
                    "part": "cv",
                    "id": cv["id"],
                    "system": name,
                    "split": cv["split"],
                    "text": ans.text,
                    "refuse": ans.refuse,
                    "citations": ans.citations,
                    "target": group,
                    "linked_skills": len(ans.linked.skills),
                    "category_match": bool(group) and dominant_category(G, group) == cv["category"],
                    "missing_mentioned": mentioned_share(ans.text, missing),
                    "grounded": (
                        len(set(ans.citations) & set(ans.allowed_ids)) / len(ans.citations)
                        if ans.citations
                        else None
                    ),
                    **_agent_record(result),
                }
            )
        print(f"  {cv['id']} xong", flush=True)
    frame = pd.DataFrame(rows)
    table = (
        frame.groupby(["split", "system"])
        .agg(
            n=("id", "size"),
            category_match=("category_match", "mean"),
            mean_linked_skills=("linked_skills", "mean"),
            missing_mentioned=("missing_mentioned", "mean"),
            citation_grounded=("grounded", "mean"),
            refusal_rate=("refuse", "mean"),
            verified_rate=("verified", "mean"),
            fallback_rate=("fallback", "mean"),
            mean_llm_calls=("llm_calls", "mean"),
            mean_seconds=("seconds", "mean"),
        )
        .reset_index()
    )
    return table, rows


def ablation() -> None:
    from career_advisor.agents.workflow import MultiAgent, SingleAgent

    graph, no_resolution = _load_graph("graph.pkl"), _load_graph("graph_no_resolution.pkl")
    client = make_client()
    single, multi = SingleAgent(client, graph), MultiAgent(client, graph)

    print("Hỏi đáp: 1 agent, 4 agent, 4 agent trên đồ thị không gộp tên")
    qa_table, qa_rows = _qa_ablation(
        {
            "single_agent": single,
            "multi_agent": multi,
            "multi_agent_no_resolution": MultiAgent(client, no_resolution),
        }
    )
    print(qa_table.round(3).to_string(index=False))
    print("Lộ trình từ CV: 1 agent, 4 agent")
    cv_table, cv_rows = _cv_ablation({"single_agent": single, "multi_agent": multi}, graph)
    print(cv_table.round(3).to_string(index=False))

    vector = json.loads((RESULTS / "qa_kg_vs_vector.json").read_text())
    result = {
        **_meta(client),
        "llm_calls_uncached": client.calls,
        "kg_vs_vector": {"source": "qa_kg_vs_vector.json", "rows": vector["rows"]},
        "single_vs_multi_agent": {
            "qa": qa_table[qa_table["system"] != "multi_agent_no_resolution"].to_dict("records"),
            "cv_plan": cv_table.to_dict("records"),
        },
        "resolution": {
            "qa": qa_table[qa_table["system"] != "single_agent"].to_dict("records"),
            "note": "multi_agent: graph.pkl (có gộp tên); multi_agent_no_resolution: graph_no_resolution.pkl",
        },
    }
    (RESULTS / "ablation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    (RESULTS / "ablation_answers.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in qa_rows + cv_rows),
        encoding="utf-8",
    )


def latency() -> None:
    """Đo thời gian luồng 4 agent trên câu hỏi dev và 5 CV dev, với cache LLM rỗng (thư mục tạm)."""
    import statistics
    import tempfile
    from pathlib import Path

    from career_advisor.agents.workflow import MultiAgent

    graph = _load_graph("graph.pkl")
    cvs = [json.loads(line) for line in (EVAL_DIR / "cvs" / "cvs.jsonl").read_text("utf-8").splitlines()]
    with tempfile.TemporaryDirectory() as tmp:
        client = make_client(cache_dir=Path(tmp))
        agent = MultiAgent(client, graph)
        agent.run(question="Lương trung vị của kế toán tổng hợp là bao nhiêu?")  # nạp model vào bộ nhớ
        runs = {
            "question": [agent.run(question=q["question"]) for q in load_questions() if q["split"] == "dev"],
            "cv_plan": [agent.run(cv_text=cv["text"]) for cv in cvs if cv["split"] == "dev"][:5],
        }
    result = {**_meta(client), "limit_seconds": 20}
    for name, results in runs.items():
        seconds = [r.seconds for r in results if r.llm_calls]
        result[name] = {
            "n": len(results),
            "n_with_llm": len(seconds),
            "median": statistics.median(seconds),
            "max": max(seconds),
            "share_within_limit": sum(s <= 20 for s in seconds) / len(seconds),
            "mean_llm_calls": statistics.mean(r.llm_calls for r in results),
        }
        print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in result[name].items()})
    (RESULTS / "latency.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands = {
        "er-thresholds": er_thresholds,
        "entity-resolution": entity_resolution,
        "retrieval": retrieval,
        "kappa": kappa,
        "kg-rag": kg_rag,
        "ablation": ablation,
        "latency": latency,
    }
    commands[sys.argv[1]]()
