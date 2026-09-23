import pandas as pd
import pytest

from career_advisor.retrieval.lexical import Bm25Searcher, TfidfSearcher, posting_documents, tokenize

DOCS = pd.Series(
    {
        10: "Kế toán tổng hợp. misa, excel. Hạch toán, lập báo cáo thuế, sử dụng phần mềm kế toán MISA",
        11: "Kế toán thuế. excel, htkk. Kê khai thuế GTGT, quyết toán thuế TNDN",
        12: "Kế toán kho. excel. Theo dõi nhập xuất tồn kho",
        20: "Lập trình viên Python. python, django, git. Phát triển API bằng Django",
        21: "Frontend Developer. react, javascript. Xây dựng giao diện web bằng ReactJS",
        22: "Kỹ sư DevOps. docker, kubernetes. Vận hành hạ tầng cloud",
        30: "Nhân viên kinh doanh. giao tiếp, đàm phán. Tìm kiếm khách hàng mới",
        31: "Nhân viên bán hàng. giao tiếp. Tư vấn sản phẩm tại cửa hàng",
        40: "Thiết kế đồ hoạ. photoshop, illustrator. Thiết kế banner quảng cáo",
        41: "Video editor. premiere, capcut. Dựng video TikTok",
    }
)
CV_ACC = "Tôi có 2 năm làm kế toán, thành thạo MISA và Excel, từng lập báo cáo thuế"


def test_tokenize_is_case_and_unicode_insensitive():
    nfd = "kế toán"  # "kế toán" dạng NFD

    assert tokenize("Kế Toán") == tokenize(nfd)


def test_tokenize_adds_syllable_bigrams():
    tokens = tokenize("phần mềm kế toán")

    assert "phần_mềm" in tokens and "kế_toán" in tokens and "kế" in tokens


@pytest.mark.parametrize("searcher_cls", [TfidfSearcher, Bm25Searcher])
def test_accounting_cv_finds_accounting_postings_first(searcher_cls):
    searcher = searcher_cls(DOCS)

    hits = searcher.search(CV_ACC, k=3)

    assert list(hits.columns) == ["posting_id", "score"]
    assert hits.posting_id.iloc[0] == 10
    assert set(hits.posting_id) <= {10, 11, 12}


@pytest.mark.parametrize("searcher_cls", [TfidfSearcher, Bm25Searcher])
def test_k_limits_results_and_scores_are_descending(searcher_cls):
    # Câu này trùng từ với 5 tin (3 tin kế toán, 2 tin bán hàng).
    hits = searcher_cls(DOCS).search("nhân viên kế toán excel giao tiếp", k=4)

    assert len(hits) == 4
    assert hits.score.is_monotonic_decreasing


@pytest.mark.parametrize("searcher_cls", [TfidfSearcher, Bm25Searcher])
def test_python_cv_finds_python_posting_first(searcher_cls):
    hits = searcher_cls(DOCS).search("lập trình python django git", k=4)

    assert hits.posting_id.iloc[0] == 20


@pytest.mark.parametrize("searcher_cls", [TfidfSearcher, Bm25Searcher])
def test_query_without_known_words_returns_nothing(searcher_cls):
    assert searcher_cls(DOCS).search("xyzzy qwerty", k=5).empty


def test_posting_documents_join_title_skills_and_text():
    postings = pd.DataFrame(
        {"job_title": ["Kế toán"], "requirements_text": ["Thành thạo MISA"], "description": ["Hạch toán"]},
        index=pd.Index([7], name="posting_id"),
    )
    skills = pd.DataFrame({"posting_id": [7, 7], "skill": ["misa", "excel"]})

    docs = posting_documents(postings, skills)

    assert docs[7] == "Kế toán. excel, misa. Thành thạo MISA. Hạch toán"
