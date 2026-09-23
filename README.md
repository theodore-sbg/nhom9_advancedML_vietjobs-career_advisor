# Trợ lý tư vấn nghề nghiệp từ tin tuyển dụng VietJobs

Đồ án môn Máy học nâng cao. Đề tài dùng KG-RAG và hệ đa tác tử trên bộ dữ liệu VietJobs.
Bản đề xuất đầy đủ: [de-an-tu-van-nghe-nghiep-vietjobs.md](de-an-tu-van-nghe-nghiep-vietjobs.md).

## Cài đặt

Cần Python 3.11 hoặc 3.12. Chưa dùng 3.14 vì PyTorch và sentence-transformers chưa hỗ trợ tốt.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Tải dữ liệu

```bash
.venv/bin/python scripts/download_data.py
```

Script tải `VietJobs.csv` (103 MB) về `data/raw/` và kiểm tra SHA-256.
Phiên bản dữ liệu được ghim theo commit trên Hugging Face, xem `src/career_advisor/data.py`.
Thư mục `data/raw/` không đưa vào git.

## Chạy test

```bash
.venv/bin/pytest
```

## Cấu trúc

```
src/career_advisor/   mã nguồn chính
scripts/              script chạy từng bước
tests/                test
data/raw/             dữ liệu gốc (không vào git)
```

## Dữ liệu

- Nguồn: Hugging Face `dinhieufam/VietJobs` (VinUniversity, LREC 2026).
- Giấy phép: bài báo ghi CC BY 4.0. Thẻ dữ liệu trên Hugging Face chưa ghi giấy phép.
- 48.092 tin đăng trên TopCV, từ tháng 7 đến tháng 10/2025. Không đại diện cho toàn bộ thị trường.
- Mã tin (`posting_id`) là số thứ tự dòng trong CSV, bắt đầu từ 0.
