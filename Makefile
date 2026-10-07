# Các lệnh kiểm chất lượng code: lint, định dạng, kiểm phần nền trên diff, test.
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
