# Database connection — Neon Postgres (production) or local SQLite (fallback)
# Guest login is memory-first; guest users are upserted into Postgres before
# document/chat writes so FK constraints succeed on the same DB as uploads.

from __future__ import annotations

import contextvars
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Set True to force SQLite even if DATABASE_URL exists (Neon unreachable).
_force_sqlite = False
_postgres_checked = False
# ContextVar so async routes see the same flag as sync Depends (thread-local did not).
_prefer_sqlite: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prefer_sqlite", default=False
)


def database_url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def sqlite_path() -> Path:
    explicit = (os.getenv("SQLITE_PATH") or "").strip()
    if explicit:
        return Path(explicit)
    # Render's container FS can be awkward; /tmp is always writable.
    if os.getenv("RENDER"):
        return Path("/tmp/genrag.db")
    return ROOT_DIR / "data" / "genrag.db"


def prefer_sqlite_for_request(enabled: bool = True) -> None:
    """Prefer SQLite for this request context (guests / fallbacks)."""
    _prefer_sqlite.set(bool(enabled))


def use_postgres() -> bool:
    if _prefer_sqlite.get():
        return False
    if _force_sqlite:
        return False
    url = database_url().lower()
    return url.startswith("postgres://") or url.startswith("postgresql://")


def postgres_configured() -> bool:
    """True when DATABASE_URL points at Postgres (ignores prefer_sqlite / force flags)."""
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


def probe_postgres(timeout_sec: float = 3.0) -> bool:
    """Return True if Postgres accepts a connection within timeout; else force SQLite."""
    global _postgres_checked
    if _postgres_checked:
        return use_postgres()
    _postgres_checked = True

    url = database_url().lower()
    if not (url.startswith("postgres://") or url.startswith("postgresql://")):
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
    thread.join(timeout_sec + 0.8)
    if thread.is_alive() or not result["ok"]:
        force_sqlite(result["err"] or "connection timed out")
        return False
    return True


def open_sqlite() -> sqlite3.Connection:
    path = sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def postgres_connection(connect_timeout: int = 5) -> Iterator[Any]:
    """Always open Postgres from DATABASE_URL (ignores prefer_sqlite / force_sqlite)."""
    if not postgres_configured():
        raise RuntimeError("DATABASE_URL is not a Postgres URL")
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(
        _pg_url(),
        row_factory=dict_row,
        connect_timeout=connect_timeout,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
