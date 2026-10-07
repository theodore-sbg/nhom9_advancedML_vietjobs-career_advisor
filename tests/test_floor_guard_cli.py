import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "floor_guard.py"
NOQA = "# " + "noqa"  # ghép chuỗi để chính tệp test này không bị bộ kiểm bắt


def _repo(tmp_path: Path) -> Path:
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    (tmp_path / "ok.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


def test_untracked_file_with_vietnamese_name_is_checked(tmp_path):
    repo = _repo(tmp_path)
    (repo / "thử nghiệm.py").write_text(f"import os  {NOQA}\n")

    result = _run(repo)

    assert result.returncode == 1
    assert "silenced-checker" in result.stderr
    assert "thử nghiệm.py" in result.stderr


def test_tracked_change_in_vietnamese_named_file_is_checked(tmp_path):
    repo = _repo(tmp_path)
    (repo / "thử.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "thêm"], cwd=repo, check=True)
    (repo / "thử.py").write_text(f"x = 1  {NOQA}\n")

    assert _run(repo).returncode == 1


def test_clean_repo_exits_zero(tmp_path):
    assert _run(_repo(tmp_path)).returncode == 0


def test_unknown_base_exits_two_not_zero(tmp_path):
    assert _run(_repo(tmp_path), "--base", "khong-co-mốc-nay").returncode == 2
