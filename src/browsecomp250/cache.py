from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from .util import canonical_sha256, utc_now_iso


class SQLiteCache:
    """Small process-safe cache used for search responses and fetched pages."""

    def __init__(self, path: Path, namespace: str):
        self.path = path
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, cache_key)
                )
                """
            )

    @staticmethod
    def key_for(request: Any) -> str:
        return canonical_sha256(request)

    def get(self, request: Any) -> Any | None:
        key = self.key_for(request)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response_json FROM cache_entries WHERE namespace=? AND cache_key=?",
                (self.namespace, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, request: Any, response: Any) -> str:
        key = self.key_for(request)
        request_json = json.dumps(request, sort_keys=True, ensure_ascii=False)
        response_json = json.dumps(response, sort_keys=True, ensure_ascii=False)
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO cache_entries(namespace, cache_key, request_json, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    request_json=excluded.request_json,
                    response_json=excluded.response_json,
                    created_at=excluded.created_at
                """,
                (self.namespace, key, request_json, response_json, utc_now_iso()),
            )
        return key

    def count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE namespace=?", (self.namespace,)
            ).fetchone()
        return int(row[0]) if row else 0
