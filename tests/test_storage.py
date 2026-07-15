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


def test_run_lock_allows_mutable_cache_snapshot_change_on_resume(tmp_path: Path) -> None:
    storage = RunStorage(tmp_path / "run")
    base = {
        "config": {"run": {"name": "same"}},
        "dataset": {"sha256": "dataset"},
        "subset_indices_sha256": "subset",
        "secret_fingerprints": {"model_api_key": "fingerprint"},
    }
    storage.write_lock({**base, "replay_hash": "before", "cache_state_at_start": {"size": 0}})
    storage.write_lock({**base, "replay_hash": "after", "cache_state_at_start": {"size": 99}})


def test_run_lock_rejects_changed_immutable_resume_contract(tmp_path: Path) -> None:
    storage = RunStorage(tmp_path / "run")
    base = {
        "config": {"run": {"name": "same"}},
        "dataset": {"sha256": "dataset"},
        "subset_indices_sha256": "subset",
        "secret_fingerprints": {"model_api_key": "fingerprint"},
    }
    storage.write_lock({**base, "replay_hash": "before"})
    import pytest

    with pytest.raises(FileExistsError):
        storage.write_lock(
            {
                **base,
                "config": {"run": {"name": "changed"}},
                "replay_hash": "after",
            }
        )
