import numpy as np
import pytest

from career_advisor.embeddings import encode_in_chunks


class FakeEncoder:
    """Mã hoá giả: vector [độ dài chuỗi, 1], chuẩn hoá. Đếm số chuỗi đã mã hoá."""

    def __init__(self):
        self.encoded = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded += len(texts)
        v = np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


TEXTS = [f"văn bản {'x' * k}" for k in range(10)]


def test_chunked_result_equals_encoding_everything_at_once(tmp_path):
    result = encode_in_chunks(TEXTS, tmp_path, FakeEncoder(), chunk_size=3)

    assert np.array_equal(result, FakeEncoder().encode(TEXTS))


def test_second_run_reuses_saved_chunks(tmp_path):
    encode_in_chunks(TEXTS, tmp_path, FakeEncoder(), chunk_size=3)
    encoder = FakeEncoder()

    encode_in_chunks(TEXTS, tmp_path, encoder, chunk_size=3)

    assert encoder.encoded == 0


def test_missing_chunk_is_the_only_one_recomputed(tmp_path):
    encode_in_chunks(TEXTS, tmp_path, FakeEncoder(), chunk_size=3)
    (tmp_path / "00001.npy").unlink()
    encoder = FakeEncoder()

    result = encode_in_chunks(TEXTS, tmp_path, encoder, chunk_size=3)

    assert encoder.encoded == 3
    assert np.array_equal(result, FakeEncoder().encode(TEXTS))


def test_changed_input_is_refused_instead_of_mixing_old_vectors(tmp_path):
    encode_in_chunks(TEXTS, tmp_path, FakeEncoder(), chunk_size=3)

    with pytest.raises(ValueError, match="khác"):
        encode_in_chunks([*TEXTS, "mới"], tmp_path, FakeEncoder(), chunk_size=3)
