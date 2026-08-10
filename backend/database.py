"""SQLite persistence for conversations. Memory and RAG stores added in later phases."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "genrag.db"

HISTORY_LIMIT = 10  # Last N message pairs sent to the LLM (truncation strategy)


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            -- Placeholder for Phase 9 (memory)
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence INTEGER DEFAULT 80,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )


def ensure_conversation(conversation_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id) VALUES (?)",
            (conversation_id,),
        )


def add_message(conversation_id: str, role: str, content: str) -> None:
    ensure_conversation(conversation_id)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def get_recent_history(conversation_id: str, limit: int = HISTORY_LIMIT * 2) -> list[dict]:
    """Return the most recent messages as OpenAI-style {role, content} dicts."""
    ensure_conversation(conversation_id)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    # Oldest first for the LLM
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_conversation(conversation_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
