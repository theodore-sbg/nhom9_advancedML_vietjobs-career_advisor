# Các lệnh kiểm theo CONSTRAINTS.md. CONSTRAINTS.md là nguồn chuẩn; lệnh ở đây phải khớp với nó.
PY := .venv/bin/python

.PHONY: check-fast check-task check-full

check-fast:  ## sau mỗi lần sửa, vài giây
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	$(PY) scripts/floor_guard.py

check-task: check-fast  ## trước khi báo xong task, dưới 90 giây
	.venv/bin/pytest -q

check-full: check-task  ## trước khi commit hoặc nộp bài; repo chưa có CI nên đây là cổng cuối
	$(PY) scripts/floor_guard.py --base $${BASE:-HEAD}
