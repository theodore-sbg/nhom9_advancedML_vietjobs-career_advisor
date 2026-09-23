"""Tải và đọc bộ dữ liệu VietJobs (Hugging Face `dinhieufam/VietJobs`)."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "VietJobs.csv"

# Ghim đúng một phiên bản để kết quả chạy lại được.
# Nếu tác giả cập nhật bộ dữ liệu, phải đổi cả 3 giá trị này và đo lại.
REPO_ID = "dinhieufam/VietJobs"
REVISION = "ea140511b77935704e93d21c2973b72f46d48902"
CSV_SHA256 = "85862b06fda4e814fe0c1d8622f173d189c92758345f77df16d1232d0c49d477"
CSV_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/VietJobs.csv"

EXPECTED_ROWS = 48_092


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path = RAW_CSV, url: str = CSV_URL, sha256: str = CSV_SHA256) -> Path:
    """Tải CSV về `dest` và kiểm tra SHA-256. Bỏ qua nếu tệp đã có và đúng mã băm."""
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    actual = sha256_of(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ValueError(f"Sai SHA-256: mong đợi {sha256}, nhận {actual}")
    tmp.replace(dest)
    return dest


def load_postings(path: Path = RAW_CSV) -> pd.DataFrame:
    """Đọc toàn bộ tin tuyển dụng. Chỉ số dòng là mã tin dùng làm dẫn chứng."""
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy: python scripts/download_data.py")
    df = pd.read_csv(path)
    df.index.name = "posting_id"
    return df
