from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet

from browsecomp250.run.storage import RunStorage


def test_transcript_encryption(monkeypatch, tmp_path: Path) -> None:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("BC250_ARTIFACT_FERNET_KEY", key)
    storage = RunStorage(tmp_path / "run")
    path = storage.write_transcript("item", 1, {"question": "private question"})
    assert path.suffix == ".fernet"
    assert b"private question" not in path.read_bytes()
    plaintext = Fernet(key.encode("ascii")).decrypt(path.read_bytes())
    assert b"private question" in plaintext


def test_run_lock_rejects_different_replay_hash(tmp_path: Path) -> None:
    storage = RunStorage(tmp_path / "run")
    storage.write_lock({"replay_hash": "a"})
    storage.write_lock({"replay_hash": "a"})
    import pytest

    with pytest.raises(FileExistsError):
        storage.write_lock({"replay_hash": "b"})
