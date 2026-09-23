import hashlib

import pytest

from career_advisor import data


def test_download_accepts_matching_hash(tmp_path):
    src = tmp_path / "src.csv"
    src.write_bytes(b"a,b\n1,2\n")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "out" / "x.csv"

    data.download(dest, url=src.as_uri(), sha256=digest)

    assert dest.read_bytes() == b"a,b\n1,2\n"


def test_download_rejects_wrong_hash_and_leaves_no_file(tmp_path):
    src = tmp_path / "src.csv"
    src.write_bytes(b"a,b\n1,2\n")
    dest = tmp_path / "x.csv"

    with pytest.raises(ValueError, match="SHA-256"):
        data.download(dest, url=src.as_uri(), sha256="0" * 64)

    assert list(tmp_path.iterdir()) == [src]


@pytest.mark.skipif(not data.RAW_CSV.exists(), reason="chưa tải dữ liệu")
def test_real_dataset_shape():
    df = data.load_postings()
    assert len(df) == data.EXPECTED_ROWS
    assert df.shape[1] == 18
    assert df.index.name == "posting_id"
