# Database connection — Neon Postgres (production) or local SQLite (fallback)
# Guest sessions always use SQLite so Neon cold-starts never block login.

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "genrag.db"
load_dotenv(ROOT_DIR / ".env")

# Set True to force SQLite even if DATABASE_URL exists (Neon unreachable).
_force_sqlite = False
_postgres_checked = False
_tls = threading.local()


def database_url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def prefer_sqlite_for_request(enabled: bool = True) -> None:
    """Route this worker-thread's get_connection() calls to SQLite (guest requests)."""
    _tls.prefer_sqlite = bool(enabled)


def use_postgres() -> bool:
    if getattr(_tls, "prefer_sqlite", False):
        return False
    if _force_sqlite:
        return False
    url = database_url().lower()
    return url.startswith("postgres://") or url.startswith("postgresql://")


def _pg_url() -> str:
    """psycopg wants postgresql:// not postgres://"""
    url = database_url()
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def force_sqlite(reason: str = "") -> None:
    global _force_sqlite
    _force_sqlite = True
    msg = "GenRAG: Neon/Postgres unavailable — using local SQLite."
    if reason:
        msg += f" ({reason})"
    print(msg, flush=True)


def probe_postgres(timeout_sec: float = 4.0) -> bool:
    """Return True if Postgres accepts a connection within timeout; else force SQLite."""
    global _postgres_checked
    if _postgres_checked:
        return use_postgres()
    _postgres_checked = True

    if not (database_url().lower().startswith("postgres://") or database_url().lower().startswith("postgresql://")):
        return False
    if _force_sqlite:
        return False

    result: dict[str, Any] = {"ok": False, "err": ""}

    def _connect() -> None:
        try:
            import psycopg

            conn = psycopg.connect(_pg_url(), connect_timeout=max(1, int(timeout_sec)))
            conn.execute("SELECT 1")
            conn.close()
            result["ok"] = True
        except Exception as exc:
            result["err"] = str(exc)

    thread = threading.Thread(target=_connect, daemon=True)
    thread.start()
    thread.join(timeout_sec + 1.0)
    if thread.is_alive() or not result["ok"]:
        force_sqlite(result["err"] or "connection timed out")
        return False
    return True


def open_sqlite() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection() -> Iterator[Any]:
    if use_postgres():
        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(
            _pg_url(),
            row_factory=dict_row,
            connect_timeout=3,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = open_sqlite()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def q(sql: str) -> str:
    """Translate %s-style SQL for SQLite (?)."""
    if use_postgres():
        return sql
    return sql.replace("%s", "?")
