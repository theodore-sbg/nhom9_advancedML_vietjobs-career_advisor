"""Gom các tệp `eval/results/*.json` thành bảng Markdown cho báo cáo (`docs/results.md`).

Chỉ đọc và định dạng, không tính lại chỉ số. Mỗi bảng ghi commit đã sinh ra nó, để truy ngược được.
Bảng dùng tập test, trừ khi tệp chỉ có một tập.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from career_advisor.rag.subgraph import fmt_number

Column = tuple[str, str, str]  # (tiêu đề, khoá trong dòng, kiểu: text | int | num | avg | pct | sec)
Table = tuple[str, list[dict], list[Column]]  # (tiêu đề phụ, dòng, cột)


def _cell(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if kind == "pct":
        return fmt_number(value, pct=True)
    if kind == "num":  # chỉ số 0–1 như F1, MRR, κ: giữ 3 chữ số
        return f"{value:.3f}".replace(".", ",")
    if kind == "avg":  # trung bình đếm được như số lượt, số kỹ năng, GB
        return fmt_number(value)
    if kind == "int":
        return str(int(value))
    if kind == "sec":
        return f"{fmt_number(value)} s"
    return str(value)


def md_table(rows: list[dict], cols: list[Column]) -> str:
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(row.get(key), kind) for _, key, kind in cols) + " |" for row in rows]
    return "\n".join(lines)


def _exact(x: float) -> str:
    """Ngưỡng phải in đúng như cấu hình, không làm tròn."""
    return f"{x:g}".replace(".", ",")


def _test(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("split") == "test"]


QA_COLS = [
    ("Hệ thống", "system", "text"),
    ("Nhóm câu", "group", "text"),
    ("Số câu", "n", "int"),
    ("Đúng", "accuracy", "pct"),
    ("Trích dẫn đúng", "citation_precision", "pct"),
    ("Từ chối nhầm", "false_refusal_rate", "pct"),
]


def _retrieval(d: dict) -> list[Table]:
    agreement = d["agreement"]["human_vs_llm"]
    kappa_rows = [
        {"level": "3 mức (0/1/2)", "kappa": agreement["unweighted"], "agree": agreement["exact_agreement"]},
        *(
            {"level": f"Nhị phân {name}", "kappa": v["kappa"], "agree": v["agreement"]}
            for name, v in agreement["binary"].items()
        ),
    ]
    return [
        (
            "Theo điểm của Qwen (Recall@10 và MRR dùng ngưỡng ≥ 2)",
            _test(d["rows"]),
            [
                ("Cách", "system", "text"),
                ("Recall@10", "recall@k", "pct"),
                ("MRR", "mrr", "num"),
                ("nDCG@10", "ndcg@k", "num"),
                ("Số CV", "n_cv", "int"),
            ],
        ),
        (
            "Kiểm chéo bằng nhãn người (tỷ lệ tin được chấm 2 = phù hợp)",
            d["human_check"]["test"],
            [
                ("Cách", "system", "text"),
                ("Số cặp có nhãn", "n_labeled_human", "int"),
                ("Phù hợp (người)", "share_relevant_human", "pct"),
                ("Phù hợp (Qwen)", "share_relevant_qwen", "pct"),
            ],
        ),
        (
            f"Độ khớp người–Qwen trên {agreement['n']} cặp",
            kappa_rows,
            [("Mức so", "level", "text"), ("κ", "kappa", "num"), ("Trùng khớp", "agree", "pct")],
        ),
    ]


def _entity_resolution(d: dict) -> list[Table]:
    cols = [
        ("Cách", "system", "text"),
        ("Precision", "precision", "pct"),
        ("Recall", "recall", "pct"),
        ("F1", "f1", "num"),
        ("Số cặp", "n", "int"),
    ]
    note = f"Ngưỡng tự gộp {_exact(d['auto_threshold'])}, hỏi LLM từ {_exact(d['llm_low'])}"
    return [(note, _test(d["rows"]), cols)]


def _qa(d: dict) -> list[Table]:
    return [("", _test(d["rows"]), QA_COLS)]


def _ablation(d: dict) -> list[Table]:
    agent_cols = [
        *QA_COLS[:4],
        ("Qua Verifier", "verified_rate", "pct"),
        ("Lượt LLM", "mean_llm_calls", "avg"),
    ]
    plan_cols = [
        ("Hệ thống", "system", "text"),
        ("Số CV", "n", "int"),
        ("Đúng nhóm ngành", "category_match", "pct"),
        ("Kỹ năng nối được / CV", "mean_linked_skills", "avg"),
        ("Qua Verifier", "verified_rate", "pct"),
        ("Thời gian / CV", "mean_seconds", "sec"),
    ]
    agents = d["single_vs_multi_agent"]
    return [
        ("1 agent so với 4 agent: hỏi đáp", _test(agents["qa"]), agent_cols),
        ("1 agent so với 4 agent: lộ trình từ CV", _test(agents["cv_plan"]), plan_cols),
        ("Bỏ bước gộp tên kỹ năng", _test(d["resolution"]["qa"]), QA_COLS),
    ]


def _rnn(d: dict) -> list[Table]:
    cols = [
        ("Mô hình", "model", "text"),
        ("Accuracy", "accuracy", "pct"),
        ("Macro-F1", "macro_f1", "num"),
        ("Thời gian", "seconds", "sec"),
    ]
    tables = []
    for task, title in (("category", "Nhóm ngành"), ("salary_band", "Khoảng lương")):
        result = d[task]
        rows = [{"model": m, **result[m]} for m in ("majority", "tfidf_logreg", "lstm")]
        tables.append((f"{title} (test: {result['n']['test']} tin)", rows, cols))
    return tables


def _data_quality(d: dict) -> list[Table]:
    human, llm = d["human_error_rates"], d["llm_error_rates"]
    rows = [
        {
            "field": field,
            "wrong": f"{human[field]['wrong']}/{human[field]['n']}",
            "human": human[field]["error_rate"],
            "llm": llm[field]["error_rate"],
        }
        for field in human
    ]
    cols = [
        ("Trường", "field", "text"),
        ("Sai (người kiểm)", "wrong", "text"),
        ("Tỷ lệ sai (người)", "human", "pct"),
        ("Tỷ lệ sai (Qwen chấm)", "llm", "pct"),
    ]
    title = (
        f"Người kiểm {d['n_human']} tin; Qwen chấm {d['n_llm']} tin ({d['llm_unparsed']} tin không đọc được)"
    )
    return [(title, rows, cols)]


def _latency(d: dict) -> list[Table]:
    rows = [{"task": task, **d[task]} for task in ("question", "cv_plan")]
    cols = [
        ("Tác vụ", "task", "text"),
        ("Số lượt", "n", "int"),
        ("Trung vị", "median", "sec"),
        ("Lâu nhất", "max", "sec"),
        (f"Trong {d['limit_seconds']} s", "share_within_limit", "pct"),
    ]
    return [("", rows, cols)]


def _llm_benchmark(d: dict) -> list[Table]:
    cols = [
        ("Model", "model", "text"),
        ("F1 gộp tên", "f1", "num"),
        ("Giây / lượt", "seconds_per_call", "sec"),
        ("RAM (GB)", "ram_gb", "avg"),
        ("JSON hỏng", "broken_json_rate", "pct"),
    ]
    return [("", list(d.values()), cols)]


SECTIONS: list[tuple[str, str, Callable[[dict], list[Table]]]] = [
    ("Truy xuất CV → tin", "retrieval", _retrieval),
    ("Gộp tên kỹ năng", "entity_resolution", _entity_resolution),
    ("Hỏi đáp: KG-RAG so với vector RAG", "qa_kg_vs_vector", _qa),
    ("Ablation", "ablation", _ablation),
    ("Phân loại văn bản: BiLSTM so với TF-IDF + hồi quy logistic", "rnn", _rnn),
    ("Chất lượng dữ liệu", "data_quality", _data_quality),
    ("Độ trễ", "latency", _latency),
    ("Chọn model LLM", "llm_benchmark", _llm_benchmark),
]


def build_report(results: dict[str, dict]) -> str:
    """`results`: tên tệp (không đuôi) → nội dung JSON đã đọc."""
    parts = [
        "# Kết quả thực nghiệm",
        "",
        "Tệp này sinh tự động bằng `scripts/summarize_results.py` từ `eval/results/`. Không sửa tay.",
        "Số liệu trên tập test, trừ khi ghi khác.",
    ]
    for title, name, tables in SECTIONS:
        parts += ["", f"## {title}", ""]
        data = results.get(name)
        if data is None:
            parts.append(f"_(chưa có `eval/results/{name}.json`)_")
            continue
        source = f"Nguồn: `eval/results/{name}.json`"
        parts.append(source + (f", commit `{data['commit']}`." if "commit" in data else "."))
        for subtitle, rows, cols in tables(data):
            parts += ["", f"**{subtitle}**", ""] if subtitle else [""]
            parts.append(md_table(rows, cols))
    return "\n".join(parts) + "\n"
