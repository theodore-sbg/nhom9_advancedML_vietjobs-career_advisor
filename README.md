# Trợ lý tư vấn nghề nghiệp từ tin tuyển dụng VietJobs

Đồ án môn Máy học nâng cao. Đề tài dùng KG-RAG và hệ đa tác tử trên bộ dữ liệu VietJobs.

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

## Cấu hình LLM

Tạo tệp `.env` (không đưa vào git):

```
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:9b
```

Model chạy cục bộ qua [Ollama](https://ollama.com), nên CV không rời khỏi máy. Cần khoảng 6 GB RAM cho model.
Cài model bằng `ollama pull qwen3.5:9b`, rồi kiểm tra bằng `.venv/bin/python scripts/check_llm.py`.
Kết quả LLM được cache trong `data/cache/llm/`, nên chạy lại cho cùng kết quả mà không gọi lại model.

## Chạy pipeline

Mỗi bước đọc đầu ra của bước trước và ghi vào `data/processed/`.

```bash
.venv/bin/python scripts/clean_data.py           # làm sạch lương, kinh nghiệm, bằng cấp, tỉnh; chia train/val/test
.venv/bin/python scripts/embed.py skills         # bge-m3 cho kỹ năng, khoảng 2 phút
.venv/bin/python scripts/resolve_skills.py       # gộp tên kỹ năng: luật → embedding → LLM
.venv/bin/python scripts/embed.py postings       # bge-m3 cho 48 nghìn tin, khoảng 60 phút
.venv/bin/python scripts/compute_stats.py        # χ², PMI, tỷ lệ kỹ năng theo nhóm
.venv/bin/python scripts/build_graph.py          # đồ thị tri thức → graph.pkl
.venv/bin/python scripts/build_graph.py --no-resolution   # đồ thị không gộp tên, cho ablation
.venv/bin/python scripts/train_rnn.py            # BiLSTM và TF-IDF + hồi quy logistic, khoảng 60 phút
```

## Chạy giao diện demo

```bash
ollama serve
.venv/bin/streamlit run app/streamlit_app.py
```

Mở http://localhost:8501, dán CV rồi bấm **Phân tích CV**. Một CV mới mất khoảng 40–60 giây trên laptop;
một câu hỏi khoảng 6 giây. Nạp sẵn model trước khi demo để lượt đầu không phải chờ:

```bash
curl localhost:11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":"60m"}'
```

## Đánh giá

```bash
.venv/bin/python scripts/evaluate.py <lệnh>      # er-thresholds, entity-resolution, retrieval, kappa,
                                                 # kg-rag, ablation, latency, data-quality, data-quality-score
.venv/bin/python scripts/summarize_results.py    # gom eval/results/*.json → docs/results.md
```

Kết quả đo nằm ở `eval/results/*.json`; chạy `scripts/summarize_results.py` để sinh bảng tổng hợp `docs/results.md`. Nhãn tay nằm ở `eval/labels/` và do người làm đồ án
gán (`label_relevance.py`, `label_data_quality.py`, `label_skill_pairs.py`).

## Test và kiểm chất lượng

```bash
make check-fast     # ruff + kiểm phần nền trên diff, vài giây
make check-task     # thêm toàn bộ pytest, khoảng 10 giây
```


## Cấu trúc

```
src/career_advisor/   mã nguồn: cleaning, graph, retrieval, rag, agents, rnn, evaluation
app/                  giao diện Streamlit
scripts/              script chạy từng bước, đánh giá, công cụ gán nhãn tay
tests/                test
eval/                 CV mẫu, bộ câu hỏi, nhãn tay, kết quả (json)
docs/                 kết quả tổng hợp, slide và script thuyết trình
data/                 dữ liệu gốc, đã xử lý và cache LLM (không vào git)
```

## Dữ liệu

- Nguồn: Hugging Face `dinhieufam/VietJobs` (VinUniversity, LREC 2026).
- Giấy phép: bài báo ghi CC BY 4.0. Thẻ dữ liệu trên Hugging Face chưa ghi giấy phép.
- 48.092 tin đăng trên TopCV, từ tháng 7 đến tháng 10/2025. Không đại diện cho toàn bộ thị trường.
- Mã tin (`posting_id`) là số thứ tự dòng trong CSV, bắt đầu từ 0.
