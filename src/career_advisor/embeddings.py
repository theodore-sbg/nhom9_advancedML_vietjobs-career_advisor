"""Mã hoá văn bản bằng bge-m3, lưu theo khối để chạy lại được khi bị ngắt giữa chừng."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 512  # mô tả tin dài hơn thì cắt, để 48 nghìn tin mã hoá xong trong thời gian chấp nhận được
BATCH_SIZE = 32
CHUNK_SIZE = 2048


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class BgeM3Encoder:
    """bge-m3 qua sentence-transformers. Vector đã chuẩn hoá độ dài 1, nên tích vô hướng là cosine."""

    def __init__(self, batch_size: int = BATCH_SIZE) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def _fingerprint(texts: list[str], chunk_size: int) -> str:
    h = hashlib.sha256(str(chunk_size).encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def encode_in_chunks(
    texts: list[str], out_dir: Path, encoder: Encoder, chunk_size: int = CHUNK_SIZE
) -> np.ndarray:
    """Mã hoá `texts`, lưu mỗi khối thành `out_dir/00000.npy`… Khối đã có thì dùng lại.

    Nếu `texts` khác lần chạy trước thì báo lỗi, để không trộn vector cũ với dữ liệu mới.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    fingerprint = _fingerprint(texts, chunk_size)
    if meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError(f"Dữ liệu đầu vào khác lần mã hoá trước. Xoá {out_dir} rồi chạy lại.")
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "n": len(texts)}))

    chunks = []
    for k, start in enumerate(range(0, len(texts), chunk_size)):
        path = out_dir / f"{k:05d}.npy"
        if path.exists():
            chunks.append(np.load(path))
            continue
        vectors = encoder.encode(texts[start : start + chunk_size])
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, vectors)
        tmp.replace(path)
        chunks.append(vectors)
    return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
