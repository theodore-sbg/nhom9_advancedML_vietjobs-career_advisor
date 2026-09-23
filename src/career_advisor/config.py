"""Hằng số dùng chung và đọc cấu hình từ `.env`."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = ROOT / "eval"
ENV_FILE = ROOT / ".env"

SEED = 42


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Đọc tệp `.env` dạng KEY=VALUE. Bỏ dòng trống, dòng `#` và dấu nháy bao quanh giá trị."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def settings() -> dict[str, str]:
    """Cấu hình hiệu lực: biến môi trường ghi đè giá trị trong `.env`."""
    return {**load_env(), **os.environ}
