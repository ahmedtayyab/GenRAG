# SQLite — conversations, messages, memories, documents

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "genrag.db"

HISTORY_LIMIT = 10  # last N turns sent to LLM (truncation)


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

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence INTEGER DEFAULT 80,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                extracted_preview TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )


def ensure_conversation(conversation_id: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO conversations (id) VALUES (?)", (conversation_id,))


def add_message(conversation_id: str, role: str, content: str) -> None:
    ensure_conversation(conversation_id)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def get_recent_history(conversation_id: str, limit: int = HISTORY_LIMIT * 2) -> list[dict]:
    ensure_conversation(conversation_id)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_conversation(conversation_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))


def add_memory(mem_id: str, text: str, category: str, confidence: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO memories (id, text, category, confidence) VALUES (?, ?, ?, ?)",
            (mem_id, text, category, confidence),
        )


def get_all_memories() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text, category, confidence FROM memories ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_memory(mem_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        return cur.rowcount > 0


def save_document(doc_id: str, filename: str, page_count: int, chunk_count: int, preview: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents (id, filename, page_count, chunk_count, extracted_preview)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, filename, page_count, chunk_count, preview[:2000]),
        )


def list_documents() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, page_count, chunk_count, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(doc_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, filename, page_count, chunk_count, extracted_preview, created_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0
