"""Tải bộ dữ liệu VietJobs về data/raw/ và in vài số liệu để kiểm tra."""

from career_advisor.data import EXPECTED_ROWS, download, load_postings

if __name__ == "__main__":
    path = download()
    df = load_postings(path)
    print(f"Đã lưu: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"Số tin: {len(df):,} (mong đợi {EXPECTED_ROWS:,})")
    print(f"Số cột: {df.shape[1]}")
    print("Các cột:", ", ".join(df.columns))
