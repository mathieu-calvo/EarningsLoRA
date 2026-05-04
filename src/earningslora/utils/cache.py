"""Tiny SQLite-backed cache for LLM calls.

Keyed by a deterministic hash of the inputs. Survives across runs so that
re-running `scripts/evaluate.py` doesn't re-spend free-tier quota. No locking
needed — eval is single-process. Same shape as `Lexico/src/lexico/cache/sqlite_cache.py`
just trimmed down for this use case.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def _hash_key(namespace: str, payload: Any) -> str:
    blob = json.dumps([namespace, payload], sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class JsonCache:
    """SQLite-backed cache mapping deterministic input hash → JSON value."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, namespace TEXT, value TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, namespace: str, payload: Any) -> Any | None:
        key = _hash_key(namespace, payload)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        key = _hash_key(namespace, payload)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, namespace, value) VALUES (?, ?, ?)",
                (key, namespace, json.dumps(value, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def __contains__(self, item: tuple[str, Any]) -> bool:
        ns, payload = item
        return self.get(ns, payload) is not None
