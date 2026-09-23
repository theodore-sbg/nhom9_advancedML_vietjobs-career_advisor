import pandas as pd
import pytest

from career_advisor.cleaning.titles import add_title_columns, normalize_title


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Kế Toán Tổng Hợp (Lương 12-15tr)", "kế toán tổng hợp"),
        ("Kế Toán Tổng Hợp", "kế toán tổng hợp"),
        # "nhân viên" đứng đầu không mang nghĩa nghề: "nhân viên kế toán" và "kế toán" là một.
        ("Nhân Viên Kế Toán", "kế toán"),
        ("Nhân Viên Kinh Doanh", "kinh doanh"),
        # "chuyên viên", "trưởng phòng"… là cấp bậc, giữ lại.
        ("Chuyên Viên Pháp Lý", "chuyên viên pháp lý"),
        # Đuôi sau " - " là mô tả thêm.
        ("Kỹ Sư Điện Mặt Trời - Từ 1 Năm Kinh Nghiệm", "kỹ sư điện mặt trời"),
        ("Kỹ Sư Môi Trường - Đi Làm Ngay", "kỹ sư môi trường"),
        ("Giám Đốc Công Ty – Hệ Thống Chuỗi Cửa Hàng F&B", "giám đốc công ty"),
        # Lương, thu nhập lẫn trong tên.
        ("Video Youtube Editor (Thu Nhập Upto 20 Triệu/Tháng)", "video youtube editor"),
        ("Thực Tập Sinh Digital Marketing Có Hỗ Trợ Lương", "thực tập sinh digital marketing"),
        ("Thực Tập Sinh Marketing Có Lương", "thực tập sinh marketing"),
        ("Nhân Viên Kinh Doanh Thu Nhập 20 Triệu", "kinh doanh"),
        # Nhiều vai trò: lấy vai trò đầu.
        ("Nhân Viên Sales/Tư Vấn Tour, Combo Du Lịch", "sales"),
        ("Nhân Viên Kinh Doanh, Sales/Telesales", "kinh doanh"),
        # Yêu cầu ngoại ngữ, nơi làm, hình thức làm việc.
        ("Nhân Viên MUA HÀNG Giao Tiếp Tiếng Anh Tốt", "mua hàng"),
        ("Giáo Viên Tiếng Anh", "giáo viên tiếng anh"),
        ("Nhân Viên Kho Tại Bình Dương", "kho"),
        ("Giáo Viên Tiếng Anh Part-Time", "giáo viên tiếng anh"),
        # Không dùng giới tính.
        ("Nữ Idol Livestream", "idol livestream"),
        ("Lễ Tân Nữ", "lễ tân"),
        # Dấu thanh và khoảng trắng.
        ("Thiết Kế Đồ Họa", "thiết kế đồ hoạ"),
        ("  Nhân   Viên  Kho ", "kho"),
        ("Nhân Viên", "nhân viên"),
        ("Sale Admin", "sales admin"),
        ("Nhân Viên Sale Ô Tô", "sales ô tô"),
        ("Wholesale Manager", "wholesale manager"),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def _postings() -> pd.DataFrame:
    titles = ["Kế Toán Tổng Hợp"] * 3 + ["Nhân Viên Kế Toán Tổng Hợp (Lương 15tr)"] * 2 + ["Nhân Viên Kho"]
    df = pd.DataFrame({"job_title": titles})
    df.index.name = "posting_id"
    return df


def test_add_title_columns_links_only_frequent_titles_to_a_node():
    out = add_title_columns(_postings(), min_postings=4)

    assert out["job_title_norm"].tolist() == ["kế toán tổng hợp"] * 5 + ["kho"]
    assert out["job_title_node"].tolist()[:5] == ["kế toán tổng hợp"] * 5
    assert pd.isna(out["job_title_node"].iloc[5])


def test_add_title_columns_keeps_most_common_raw_title_for_display():
    out = add_title_columns(_postings(), min_postings=4)

    assert set(out.loc[out.job_title_norm == "kế toán tổng hợp", "job_title_display"]) == {"Kế Toán Tổng Hợp"}
