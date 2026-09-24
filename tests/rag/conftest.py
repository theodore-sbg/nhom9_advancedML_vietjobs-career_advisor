import pandas as pd
import pytest

from career_advisor.graph.build import build_graph


@pytest.fixture(scope="session")
def small_graph():
    rows = []
    # 7 tin kế toán tổng hợp ở Bắc Ninh, 3 ở Hà Nội; 6 tin kinh doanh ở Hà Nội.
    for k, salary in enumerate([10, 11, 12, 12.5, 13, 14, 15]):
        rows.append(
            ("tài_chính_kế_toán", ["bắc ninh"], salary, "1_2y" if k < 5 else "3_4y", "kế toán tổng hợp")
        )
    for salary in (14, 15, 16):
        rows.append(("tài_chính_kế_toán", ["hà nội"], salary, "1_2y", "kế toán tổng hợp"))
    for salary in (8, 9, 10, 11, 12, None):
        rows.append(("kinh_doanh_bán_hàng", ["hà nội"], salary, "none", "kinh doanh"))
    # "văn phòng" là chức danh, nhưng cũng nằm trong kỹ năng "tin học văn phòng".
    for salary in (8, 9):
        rows.append(("nhân_sự_hành_chính", ["hà nội"], salary, "none", "văn phòng"))
    for salary in (9, 10):
        rows.append(("nhân_sự_hành_chính", ["hà nội"], salary, "none", "trợ lý"))
    # Chức danh dài 5 chữ, viết kèm "Nhân viên" thành 7 chữ; "nhân viên" cũng là một chức danh chung.
    rows.append(("kinh_doanh_bán_hàng", ["hà nội"], 9, "none", "kinh doanh tư vấn du lịch"))
    rows.append(("kinh_doanh_bán_hàng", ["hà nội"], 9, "none", "nhân viên"))
    rows.append(("nhân_sự_hành_chính", ["hà nội"], 9, "none", "tuyển dụng"))
    # Chức danh "kỹ thuật" nằm trong tên nhóm ngành "công nghệ thông tin kỹ thuật số".
    for salary in (20, 25):
        rows.append(("công_nghệ_thông_tin_kỹ_thuật_số", ["hà nội"], salary, "3_4y", "kỹ thuật"))
    postings = pd.DataFrame(
        rows, columns=["category", "provinces", "salary_mid", "experience_level", "job_title_node"]
    )
    postings["salary_min"] = postings["salary_mid"] - 1
    postings["salary_max"] = postings["salary_mid"] + 1
    postings["salary_band"] = postings["salary_mid"].map(lambda m: None if pd.isna(m) else "10_15m")
    postings["education_level"] = "college"
    postings["job_title_display"] = postings["job_title_node"].str.title()
    postings.index.name = "posting_id"
    skills = {
        **{i: ["excel", "misa", "phần mềm kế toán", "kế toán tổng hợp"] for i in range(0, 10, 2)},
        **{i: ["excel", "phần mềm kế toán", "cẩn thận"] for i in range(1, 10, 2)},
        **{i: ["giao tiếp", "đàm phán", "excel"] for i in range(10, 16)},
        # Node rác do dữ liệu gốc trích sai, trùng chữ thường gặp trong câu hỏi.
        **{
            i: ["trung", "nhóm", "python", "tin học văn phòng", "có kiến thức tốt về tài chính"]
            for i in (16, 17)
        },
    }
    mentions = pd.DataFrame(
        [(pid, s) for pid, ss in skills.items() for s in ss], columns=["posting_id", "skill"]
    )
    merges = pd.DataFrame(
        {"skill": ["ms excel"], "canonical": ["excel"], "tier": ["embedding"], "score": [0.95]}
    )
    names = sorted({s for ss in skills.values() for s in ss})
    kinds = ["soft" if n in ("cẩn thận", "giao tiếp", "đàm phán") else "technical" for n in names]
    skill_map = pd.DataFrame({"raw": names, "canonical": names, "kind": kinds, "tier": "rule"})
    return build_graph(
        postings,
        mentions,
        merges,
        skill_map,
        min_skill_postings=1,
        min_chi2_postings=1,
        min_pair=1,
        min_rate_with=1,
        alpha=1.0,
    )
