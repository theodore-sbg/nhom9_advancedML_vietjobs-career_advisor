import numpy as np
import pandas as pd
import pytest

from career_advisor.graph.build import build_graph
from career_advisor.retrieval.dense import DenseSearcher
from career_advisor.retrieval.hybrid import GraphSearcher, HybridSearcher, extract_skills, rrf


def _ranking(ids):
    return pd.DataFrame({"posting_id": ids, "score": np.linspace(1, 0.1, len(ids))})


def test_rrf_matches_hand_computation():
    fused = rrf([_ranking([1, 2, 3]), _ranking([2, 3, 4])], k_rrf=60, k=10)

    expected = {1: 1 / 61, 2: 1 / 62 + 1 / 61, 3: 1 / 63 + 1 / 62, 4: 1 / 63}
    assert fused.posting_id.tolist() == [2, 3, 1, 4]
    assert fused.set_index("posting_id").score.to_dict() == pytest.approx(expected)


def test_rrf_ties_break_by_posting_id():
    fused = rrf([_ranking([5]), _ranking([3])], k_rrf=60, k=10)

    assert fused.posting_id.tolist() == [3, 5]


class FakeEncoder:
    def encode(self, texts):
        v = np.array([[t.count("kế toán"), t.count("python"), 1.0] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dense_searcher_ranks_by_cosine():
    ids = np.array([10, 20, 30])
    vectors = FakeEncoder().encode(["kế toán kế toán", "python python", "khác"])

    hits = DenseSearcher(vectors, ids, FakeEncoder()).search("tôi làm kế toán", k=2)

    assert hits.posting_id.tolist()[0] == 10
    assert len(hits) == 2


class Fixed:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k=10):
        return _ranking(self.ids[:k])


def test_hybrid_fuses_its_searchers():
    hybrid = HybridSearcher([Fixed([1, 2, 3]), Fixed([2, 3, 4])], depth=3)

    assert hybrid.search("x", k=2).posting_id.tolist() == [2, 3]


def _graph(extra_merges=(), extra_mentions=()):
    postings = pd.DataFrame(
        {
            "category": ["acc", "acc", "acc", "it", "it"],
            "provinces": [["hà nội"]] * 5,
            "salary_min": [8.0] * 5,
            "salary_max": [12.0] * 5,
            "salary_mid": [10.0] * 5,
            "salary_band": ["10_15m"] * 5,
            "experience_level": ["1_2y"] * 5,
            "education_level": ["college"] * 5,
            "job_title_node": [None] * 5,
            "job_title_display": [None] * 5,
        },
        index=pd.Index(range(5), name="posting_id"),
    )
    mentions = pd.DataFrame(
        [
            (0, "misa"),
            (0, "excel"),
            (1, "excel"),
            (2, "excel"),
            (3, "python"),
            (4, "python"),
            (4, "excel"),
            *extra_mentions,
        ],
        columns=["posting_id", "skill"],
    )
    merges = pd.DataFrame(
        [("phần mềm misa", "misa", "llm", 0.9), *extra_merges],
        columns=["skill", "canonical", "tier", "score"],
    )
    skill_map = pd.DataFrame(
        {
            "raw": ["MISA", "Excel", "Python"],
            "canonical": ["misa", "excel", "python"],
            "kind": ["technical"] * 3,
            "tier": ["rule"] * 3,
        }  # fmt: skip
    )
    return build_graph(postings, mentions, merges, skill_map, min_skill_postings=1, min_chi2_postings=1,
                       min_pair=1, min_rate_with=1, alpha=1.0)  # fmt: skip


def test_extract_skills_finds_graph_skills_and_aliases_in_free_text():
    found = extract_skills(_graph(), "Tôi thành thạo Excel và phần mềm MISA, biết chút Python.")

    assert found == {"excel", "misa", "python"}


def test_graph_searcher_prefers_postings_sharing_rare_skills():
    hits = GraphSearcher(_graph()).search("Tôi dùng MISA và Excel", k=5)

    # Tin 0 có cả misa (hiếm) và excel; các tin chỉ có excel (phổ biến) xếp sau.
    assert hits.posting_id.iloc[0] == 0
    assert 3 not in set(hits.posting_id)  # tin 3 chỉ có python


def test_extract_skills_keeps_a_skill_hidden_inside_a_longer_alias():
    # "misa excel" là alias đã gộp vào "excel"; khớp cụm dài nhất sẽ làm mất "misa".
    G = _graph(extra_merges=[("misa excel", "excel", "embedding", 0.91)])

    assert extract_skills(G, "Thành thạo MISA Excel") == {"misa", "excel"}


def test_extract_skills_drops_a_fragment_of_a_longer_matched_skill():
    # Dữ liệu có node rác "lập" (trích sai) bên cạnh "lập trình".
    G = _graph(extra_mentions=[(3, "lập"), (3, "lập trình")])

    assert extract_skills(G, "Có kinh nghiệm lập trình Python") == {"lập trình", "python"}


def test_extract_skills_ignores_phrases_joined_by_a_conjunction():
    # Chuỗi ghép "excel và misa" trong dữ liệu gốc đã bị gộp vào "misa phần mềm".
    # Không được vì thế mà mất "misa".
    G = _graph(
        extra_merges=[("excel và misa", "misa phần mềm", "embedding", 0.9)],
        extra_mentions=[(0, "misa phần mềm"), (1, "misa phần mềm")],
    )

    assert extract_skills(G, "Tôi biết excel và misa") == {"excel", "misa"}


def test_graph_searcher_breaks_ties_by_posting_id_whatever_the_skill_order(monkeypatch):
    # misa ở tin 0 và 1, python ở tin 3 và 4: cùng IDF, nên bốn tin bằng điểm.
    G = _graph(extra_mentions=[(1, "misa")])
    rankings = []
    for order in (["misa", "python"], ["python", "misa"]):
        # Thứ tự duyệt một set chuỗi đổi theo PYTHONHASHSEED giữa các lần chạy; giả lập cả hai thứ tự.
        monkeypatch.setattr("career_advisor.retrieval.hybrid.extract_skills", lambda G, text, o=order: o)
        rankings.append(GraphSearcher(G).search("misa và python", k=4).posting_id.tolist())

    assert rankings == [[0, 1, 3, 4], [0, 1, 3, 4]]
