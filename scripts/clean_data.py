"""Bước data-cleaning: đọc CSV gốc, ghi các bảng sạch vào data/processed/."""

from career_advisor.cleaning.skills import build_skill_tables, parse_list
from career_advisor.cleaning.splits import make_splits
from career_advisor.config import PROCESSED_DIR
from career_advisor.data import load_postings

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    postings = load_postings()

    splits = make_splits(postings)
    splits.to_frame().to_parquet(PROCESSED_DIR / "splits.parquet")
    print("Tập chia:", splits.value_counts().reindex(["train", "val", "test"]).to_dict())

    mentions, skill_map = build_skill_tables(postings)
    mentions.to_parquet(PROCESSED_DIR / "skill_mentions.parquet")
    skill_map.to_parquet(PROCESSED_DIR / "skill_map.parquet")
    raw_technical = {s for cell in postings["technical_skills"] for s in parse_list(cell)}
    raw_soft = {s for cell in postings["soft_skills"] for s in parse_list(cell)}
    kinds = skill_map.drop_duplicates("canonical")["kind"].value_counts().to_dict()
    print(
        f"Chuỗi kỹ năng gốc: {len(raw_technical | raw_soft):,} "
        f"(chuyên môn {len(raw_technical):,}, mềm {len(raw_soft):,})"
    )
    print(f"Sau tầng luật: {skill_map.canonical.nunique():,} kỹ năng ({kinds})")
    print(f"Lượt nhắc sau khi bỏ trùng trong tin: {len(mentions):,}")
