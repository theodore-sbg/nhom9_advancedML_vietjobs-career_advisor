"""Trợ lý tư vấn nghề nghiệp từ tin tuyển dụng VietJobs.

.venv/bin/streamlit run app/streamlit_app.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from career_advisor.agents.workflow import MultiAgent
from career_advisor.app_support import answer_subgraph, match_table, skills_table, subgraph_html
from career_advisor.config import EVAL_DIR, PROCESSED_DIR
from career_advisor.evaluation.agent_eval import dominant_category
from career_advisor.graph import query as q
from career_advisor.llm import make_client
from career_advisor.rag.subgraph import fmt_number, group_node

TOP_MATCHES = 10
SEARCH_DEPTH = 300  # lấy rộng rồi lọc theo tỉnh
TOP_MISSING = 10
BY_CV = "— theo CV —"
LIMITS = (
    "Dữ liệu chỉ gồm tin đăng trên TopCV từ tháng 7 đến tháng 10/2025. 28,6% tin ghi lương thoả thuận "
    "đã bị loại khỏi thống kê lương. Mức lương chỉ để tham khảo, không phải cam kết."
)


@st.cache_resource(show_spinner="Đang nạp đồ thị và dữ liệu…")
def load_data():
    with (PROCESSED_DIR / "graph.pkl").open("rb") as f:
        graph = pickle.load(f)
    postings = pd.read_parquet(PROCESSED_DIR / "postings.parquet")
    skills = pd.read_parquet(PROCESSED_DIR / "stats" / "posting_skills.parquet")
    skills_of = skills.groupby("posting_id")["skill"].apply(list).to_dict()
    return graph, postings, skills_of


@st.cache_resource(show_spinner="Đang nạp bge-m3…")
def load_searcher(_postings: pd.DataFrame):
    from career_advisor.embeddings import BgeM3Encoder
    from career_advisor.retrieval.dense import DenseSearcher

    vectors = np.load(PROCESSED_DIR / "emb" / "postings.npy")
    return DenseSearcher(vectors, _postings.index.to_numpy(), BgeM3Encoder())


@st.cache_resource
def load_agents(_graph):
    client = make_client()
    return client, MultiAgent(client, _graph)


def sample_cv() -> str:
    cvs = [json.loads(line) for line in (EVAL_DIR / "cvs" / "cvs.jsonl").read_text("utf-8").splitlines()]
    return next(cv["text"] for cv in cvs if cv["split"] == "dev")


def names(graph, node_type: str) -> list[str]:
    return sorted(d["name"] for _, d in graph.nodes(data=True) if d.get("type") == node_type)


def show_answer(result, graph) -> None:
    answer, check = result.answer, result.verification
    st.markdown(answer.text)
    status = "✅ qua Verifier" if check.ok else "⚠️ Verifier chặn: " + "; ".join(check.reasons)
    if result.fallback:
        status = "⚠️ Viết lại vẫn không qua Verifier, nên hiện nguyên số liệu gốc"
    st.caption(
        f"{status} · {result.attempts} lần viết · {result.llm_calls} lượt LLM · "
        f"{result.seconds:.1f} giây · luồng: {' → '.join(result.trace)}"
    )
    with st.expander("Dữ kiện lấy từ đồ thị"):
        for fact in answer.facts:
            st.markdown(f"- {fact.text} " + " ".join(f"`#{pid}`" for pid in fact.sources))
    # Mở sẵn: vis.js khởi tạo trong khung đang đóng thì đo được kích thước 0 và vẽ lệch.
    with st.expander("Đồ thị con quanh câu trả lời", expanded=True):
        sub = answer_subgraph(graph, answer)
        if len(sub):
            st.iframe(subgraph_html(sub), height=500)
            st.caption(f"{sub.number_of_nodes()} node, {sub.number_of_edges()} cạnh.")
        else:
            st.write("Không có node nào để vẽ.")


def main() -> None:
    st.set_page_config(page_title="Tư vấn nghề nghiệp VietJobs", page_icon="💼", layout="wide")
    st.title("Trợ lý tư vấn nghề nghiệp từ tin tuyển dụng")
    st.warning(LIMITS)

    graph, postings, skills_of = load_data()
    client, agent = load_agents(graph)
    st.sidebar.markdown(f"**LLM:** `{client.provider}:{client.model}`")
    st.sidebar.markdown(f"**Đồ thị:** {graph.number_of_nodes():,} node, {graph.number_of_edges():,} cạnh")
    st.sidebar.markdown("**Luồng:** CV Agent → Graph Agent → Planner → Verifier")

    st.header("1. CV của bạn")
    cv_text = st.text_area("Dán CV dạng văn bản", value=sample_cv(), height=160)
    left, right = st.columns(2)
    title = left.selectbox("Nghề muốn làm", [BY_CV, *names(graph, "JobTitle")])
    province = right.selectbox("Tỉnh muốn làm", [BY_CV, *[p.title() for p in names(graph, "Province")]])
    if st.button("Phân tích CV", type="primary") and cv_text.strip():
        with st.spinner("4 agent đang làm việc…"):
            st.session_state["cv_result"] = agent.run(
                cv_text=cv_text,
                title=None if title == BY_CV else title,
                province=None if province == BY_CV else province.lower(),
            )
            st.session_state["cv_text"] = cv_text

    result = st.session_state.get("cv_result")
    if result:
        linked = result.answer.linked
        group = group_node(linked)
        st.info(
            f"**Hồ sơ đọc được:** nghề {', '.join(linked.titles or linked.categories) or 'chưa rõ'} · "
            f"tỉnh {', '.join(p.title() for p in linked.provinces) or 'chưa rõ'} · "
            f"kinh nghiệm {', '.join(linked.experience_levels) or 'chưa rõ'} · "
            f"kỹ năng: {', '.join(linked.skills) or 'chưa nối được'}"
        )
        tabs = st.tabs(["Lộ trình", "Tin phù hợp", "Kỹ năng thiếu", "Lương"])
        with tabs[0]:
            show_answer(result, graph)
        with tabs[1]:
            found = load_searcher(postings).search(st.session_state["cv_text"], k=SEARCH_DEPTH)
            ids = [int(pid) for pid in found["posting_id"]]
            if linked.provinces:
                ids = [pid for pid in ids if linked.provinces[0] in postings.at[pid, "provinces"]]
            st.table(match_table(postings, skills_of, ids[:TOP_MATCHES], linked.skills), hide_index=True)
            st.caption("Xếp bằng bge-m3 (cách truy xuất tốt nhất trên bộ đánh giá). Lý do: kỹ năng trùng.")
        with tabs[2]:
            if group:
                category = q.category_id(dominant_category(graph, group))
                missing = q.missing_skills(graph, linked.skills, group, k=TOP_MISSING)
                st.table(skills_table(missing, graph, category), hide_index=True)
                st.caption(
                    f"Tỷ lệ tin của nghề yêu cầu kỹ năng. χ²: mức gắn của kỹ năng với nhóm ngành "
                    f"{graph.nodes[category]['name'].replace('_', ' ')}; '–' là không có ý nghĩa thống kê."
                )
            else:
                st.write("Chưa xác định được nghề muốn làm.")
        with tabs[3]:
            scopes = [group, *[q.province_id(p) for p in linked.provinces]] if group else []
            scopes += [q.experience_id(e) for e in linked.experience_levels]
            rows = []
            for nodes in (scopes[:1], scopes[:2], scopes):
                summary = q.salary_summary(graph, *nodes) if nodes else None
                if summary and summary.n:
                    rows.append(
                        {
                            "Phạm vi": " · ".join(graph.nodes[n]["name"] for n in nodes),
                            "Trung vị (triệu)": fmt_number(summary.median),
                            "Khoảng giữa": f"{fmt_number(summary.q1)}–{fmt_number(summary.q3)}",
                            "Số tin có lương": summary.n,
                        }
                    )
            st.table(pd.DataFrame(rows).drop_duplicates(), hide_index=True)
            st.caption("Dưới 5 tin thì không nên dựa vào mức lương.")

    st.header("2. Hỏi đáp")
    question = st.text_input(
        "Câu hỏi", placeholder="Lương trung vị của kế toán tổng hợp ở Hà Nội là bao nhiêu?"
    )
    if st.button("Hỏi") and question.strip():
        with st.spinner("Đang tra đồ thị…"):
            st.session_state["qa_result"] = agent.run(question=question)
    if st.session_state.get("qa_result"):
        show_answer(st.session_state["qa_result"], graph)


main()
