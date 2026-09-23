"""Bước data-cleaning: đọc CSV gốc, ghi các bảng sạch vào data/processed/."""

from career_advisor.cleaning.splits import make_splits
from career_advisor.config import PROCESSED_DIR
from career_advisor.data import load_postings

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    postings = load_postings()

    splits = make_splits(postings)
    splits.to_frame().to_parquet(PROCESSED_DIR / "splits.parquet")
    print("Tập chia:", splits.value_counts().reindex(["train", "val", "test"]).to_dict())
