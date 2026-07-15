from pathlib import Path

from browsecomp250.cache import SQLiteCache


def test_sqlite_cache_roundtrip(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "cache.sqlite3", "test")
    request = {"q": "hello"}
    assert cache.get(request) is None
    cache.put(request, {"answer": 1})
    assert cache.get(request) == {"answer": 1}
    assert cache.count() == 1
