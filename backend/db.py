# Database connection — Neon Postgres (production) or local SQLite (fallback)

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "genrag.db"
load_dotenv(ROOT_DIR / ".env")


def database_url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def use_postgres() -> bool:
    url = database_url().lower()
    return url.startswith("postgres://") or url.startswith("postgresql://")


def _pg_url() -> str:
    """psycopg wants postgresql:// not postgres://"""
    url = database_url()
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


@contextmanager
def get_connection() -> Iterator[Any]:
    if use_postgres():
        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(
            _pg_url(),
            row_factory=dict_row,
            connect_timeout=15,
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
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.row_factory = sqlite3.Row
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
