# Users, sessions, conversations, messages, memories, documents (user-scoped)

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from db import get_connection, open_sqlite, probe_postgres, prefer_sqlite_for_request, q, use_postgres

HISTORY_LIMIT = 10
SESSION_DAYS = 30
EMBED_DIM = 768


def init_db() -> None:
    # Fail fast to SQLite if Neon is paused/unreachable.
    if probe_postgres(timeout_sec=3.0):
        import threading

        box: dict = {"err": None}

        def _pg_init() -> None:
            try:
                _init_postgres()
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        thread = threading.Thread(target=_pg_init, daemon=True)
        thread.start()
        thread.join(8.0)
        if thread.is_alive() or box["err"] is not None:
            from db import force_sqlite

            force_sqlite(str(box["err"] or "postgres schema init timed out"))
    # Always keep a local SQLite schema for guest sessions (never block on Neon).
    try:
        _init_sqlite()
    except Exception as exc:  # noqa: BLE001
        print(f"GenRAG: sqlite init warning ({exc})", flush=True)


def _init_postgres() -> None:
    with get_connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                google_sub TEXT UNIQUE,
                email TEXT,
                name TEXT,
                picture TEXT,
                is_guest BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence INTEGER DEFAULT 80,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                extracted_preview TEXT,
                content_hash TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page INTEGER NOT NULL DEFAULT 1,
                content TEXT NOT NULL,
                embedding vector({EMBED_DIM}),
                UNIQUE (document_id, chunk_index)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_user_hash ON documents(user_id, content_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id)"
        )


def _init_sqlite() -> None:
    conn = open_sqlite()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                google_sub TEXT UNIQUE,
                email TEXT,
                name TEXT,
                picture TEXT,
                is_guest INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
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
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence INTEGER DEFAULT 80,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                extracted_preview TEXT,
                content_hash TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        _migrate_sqlite(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_sqlite(conn) -> None:
    """Add user_id to legacy tables if upgrading an old local DB."""
    for table in ("conversations", "documents", "memories"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "user_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
    # orphan rows: attach to a synthetic local guest so old data is not lost silently
    orphan_user = "legacy-local-user"
    conn.execute(
        """
        INSERT OR IGNORE INTO users (id, is_guest, name)
        VALUES (?, 1, 'Legacy local')
        """,
        (orphan_user,),
    )
    for table in ("conversations", "documents", "memories"):
        conn.execute(
            f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
            (orphan_user,),
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_user_hash ON documents(user_id, content_hash)"
    )


def _row(row) -> dict | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# ── Users / sessions ───────────────────────────────────────────────


def create_guest_user() -> dict:
    user_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            q(
                """
                INSERT INTO users (id, is_guest, name)
                VALUES (%s, %s, %s)
                """
            ),
            (user_id, True if use_postgres() else 1, "Guest"),
        )
    return {
        "id": user_id,
        "google_sub": None,
        "email": None,
        "name": "Guest",
        "picture": None,
        "is_guest": True,
        "created_at": None,
    }


_MEM_SESSIONS: dict[str, dict] = {}


def create_guest_session_bundle() -> tuple[dict, str]:
    """Create guest user + session instantly (memory first; SQLite in background)."""
    import threading

    user_id = str(uuid4())
    session_id = str(uuid4())
    user = {
        "id": user_id,
        "google_sub": None,
        "email": None,
        "name": "Guest",
        "picture": None,
        "is_guest": True,
        "created_at": None,
    }
    _MEM_SESSIONS[session_id] = dict(user)

    def _persist() -> None:
        try:
            _init_sqlite()
            expires = _iso(_now() + timedelta(days=SESSION_DAYS))
            conn = open_sqlite()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO users (id, is_guest, name) VALUES (?, ?, ?)",
                    (user_id, 1, "Guest"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (id, user_id, expires_at) VALUES (?, ?, ?)",
                    (session_id, user_id, expires),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            print(f"GenRAG: guest sqlite persist skipped ({exc})", flush=True)

    threading.Thread(target=_persist, daemon=True).start()
    return user, session_id


def upsert_google_user(google_sub: str, email: str | None, name: str | None, picture: str | None) -> dict:
    with get_connection() as conn:
        existing = conn.execute(
            q("SELECT id FROM users WHERE google_sub = %s"),
            (google_sub,),
        ).fetchone()
        if existing:
            user_id = existing["id"] if isinstance(existing, dict) else existing[0]
            conn.execute(
                q(
                    """
                    UPDATE users
                    SET email = %s, name = %s, picture = %s, is_guest = %s
                    WHERE id = %s
                    """
                ),
                (email, name, picture, False if use_postgres() else 0, user_id),
            )
        else:
            user_id = str(uuid4())
            conn.execute(
                q(
                    """
                    INSERT INTO users (id, google_sub, email, name, picture, is_guest)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                ),
                (user_id, google_sub, email, name, picture, False if use_postgres() else 0),
            )
    return get_user(user_id)


def get_user(user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            q(
                """
                SELECT id, google_sub, email, name, picture, is_guest, created_at
                FROM users WHERE id = %s
                """
            ),
            (user_id,),
        ).fetchone()
    user = _row(row)
    if user is not None:
        user["is_guest"] = bool(user.get("is_guest"))
    return user


def create_session(user_id: str) -> str:
    session_id = str(uuid4())
    expires = _now() + timedelta(days=SESSION_DAYS)
    with get_connection() as conn:
        conn.execute(
            q(
                """
                INSERT INTO sessions (id, user_id, expires_at)
                VALUES (%s, %s, %s)
                """
            ),
            (session_id, user_id, expires if use_postgres() else _iso(expires)),
        )
    return session_id


def _session_user_from_conn(conn, session_id: str, postgres: bool) -> dict | None:
    sql = """
        SELECT u.id, u.google_sub, u.email, u.name, u.picture, u.is_guest, u.created_at,
               s.expires_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.id = {}
    """.format("%s" if postgres else "?")
    row = conn.execute(sql, (session_id,)).fetchone()
    data = _row(row)
    if not data:
        return None
    expires = data.pop("expires_at", None)
    if expires is not None:
        if isinstance(expires, str):
            try:
                expires_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            expires_dt = expires
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        if expires_dt < _now():
            conn.execute(
                "DELETE FROM sessions WHERE id = {}".format("%s" if postgres else "?"),
                (session_id,),
            )
            return None
    data["is_guest"] = bool(data.get("is_guest"))
    return data


def get_user_by_session(session_id: str) -> dict | None:
    if not session_id:
        return None

    mem = _MEM_SESSIONS.get(session_id)
    if mem:
        return dict(mem)

    # Guests live in SQLite — check it next so Neon hangs never block auth.
    try:
        conn = open_sqlite()
        try:
            data = _session_user_from_conn(conn, session_id, postgres=False)
            if data:
                conn.commit()
                _MEM_SESSIONS[session_id] = dict(data)
                return data
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"GenRAG: sqlite session lookup failed ({exc})", flush=True)

    url = (os.getenv("DATABASE_URL") or "").lower()
    if not url.startswith("postgres"):
        return None

    prefer_sqlite_for_request(False)
    if not use_postgres():
        return None

    # Hard-cap Postgres lookup so a hung Neon pooler cannot stall the UI.
    import threading

    box: dict = {"data": None}

    def _pg_lookup() -> None:
        try:
            with get_connection() as conn:
                box["data"] = _session_user_from_conn(conn, session_id, postgres=True)
        except Exception as exc:  # noqa: BLE001
            print(f"GenRAG: postgres session lookup failed ({exc})", flush=True)

    thread = threading.Thread(target=_pg_lookup, daemon=True)
    thread.start()
    thread.join(2.5)
    return box["data"]


def delete_session(session_id: str) -> None:
    if not session_id:
        return
    conn = open_sqlite()
    try:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    prefer_sqlite_for_request(False)
    url = (os.getenv("DATABASE_URL") or "").lower()
    if url.startswith("postgres") and use_postgres():
        try:
            with get_connection() as conn:
                conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        except Exception:
            pass


def claim_guest_data(guest_user_id: str, google_user_id: str) -> None:
    """Move guest-owned rows to a Google account (sign-in to keep workspace)."""
    if not guest_user_id or guest_user_id == google_user_id:
        return
    with get_connection() as conn:
        for table in ("conversations", "documents", "memories"):
            conn.execute(
                q(f"UPDATE {table} SET user_id = %s WHERE user_id = %s"),
                (google_user_id, guest_user_id),
            )
        if use_postgres():
            conn.execute(
                q("UPDATE document_chunks SET user_id = %s WHERE user_id = %s"),
                (google_user_id, guest_user_id),
            )
        conn.execute(q("DELETE FROM sessions WHERE user_id = %s"), (guest_user_id,))
        conn.execute(q("DELETE FROM users WHERE id = %s AND is_guest = %s"), (guest_user_id, True if use_postgres() else 1))


# ── Conversations ──────────────────────────────────────────────────


def ensure_conversation(conversation_id: str, user_id: str) -> bool:
    """Create conversation for user, or confirm ownership. Returns False if id belongs to someone else."""
    with get_connection() as conn:
        row = conn.execute(
            q("SELECT user_id FROM conversations WHERE id = %s"),
            (conversation_id,),
        ).fetchone()
        if row:
            owner = row["user_id"] if isinstance(row, dict) else row[0]
            return owner == user_id
        if use_postgres():
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
                """,
                (conversation_id, user_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
                """,
                (conversation_id, user_id),
            )
    return True


def user_owns_conversation(conversation_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            q("SELECT 1 AS ok FROM conversations WHERE id = %s AND user_id = %s"),
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def add_message(conversation_id: str, role: str, content: str, user_id: str) -> None:
    if not ensure_conversation(conversation_id, user_id):
        raise PermissionError("Conversation belongs to another user.")
    with get_connection() as conn:
        conn.execute(
            q("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)"),
            (conversation_id, role, content),
        )
        if use_postgres():
            conn.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s AND user_id = %s",
                (conversation_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )


def get_recent_history(conversation_id: str, user_id: str, limit: int = HISTORY_LIMIT * 2) -> list[dict]:
    if not user_owns_conversation(conversation_id, user_id):
        if not ensure_conversation(conversation_id, user_id):
            return []
    with get_connection() as conn:
        rows = conn.execute(
            q(
                """
                SELECT role, content FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC LIMIT %s
                """
            ),
            (conversation_id, limit),
        ).fetchall()
    items = [_row(r) for r in reversed(rows)]
    return [{"role": r["role"], "content": r["content"]} for r in items]


def get_full_history(conversation_id: str, user_id: str) -> list[dict]:
    if not user_owns_conversation(conversation_id, user_id):
        return []
    with get_connection() as conn:
        rows = conn.execute(
            q(
                """
                SELECT role, content FROM messages
                WHERE conversation_id = %s
                ORDER BY id ASC
                """
            ),
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in (_row(x) for x in rows)]


def count_messages(conversation_id: str, user_id: str) -> int:
    if not user_owns_conversation(conversation_id, user_id):
        return 0
    with get_connection() as conn:
        row = conn.execute(
            q("SELECT COUNT(*) AS n FROM messages WHERE conversation_id = %s"),
            (conversation_id,),
        ).fetchone()
    data = _row(row)
    return int(data["n"]) if data else 0


def get_conversation(conversation_id: str, user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            q(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations WHERE id = %s AND user_id = %s
                """
            ),
            (conversation_id, user_id),
        ).fetchone()
    return _row(row)


def set_conversation_title(conversation_id: str, title: str, user_id: str) -> None:
    if not ensure_conversation(conversation_id, user_id):
        return
    with get_connection() as conn:
        if use_postgres():
            conn.execute(
                "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s AND user_id = %s",
                (title, conversation_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
                (title, conversation_id, user_id),
            )


def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        if use_postgres():
            rows = conn.execute(
                """
                SELECT c.id,
                       COALESCE(
                         NULLIF(c.title, ''),
                         (
                           SELECT CASE
                             WHEN length(m.content) > 48 THEN left(m.content, 45) || '…'
                             ELSE m.content
                           END
                           FROM messages m
                           WHERE m.conversation_id = c.id AND m.role = 'user'
                           ORDER BY m.id ASC
                           LIMIT 1
                         ),
                         'Untitled chat'
                       ) AS title,
                       c.created_at,
                       c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                FROM conversations c
                WHERE c.user_id = %s
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
                ORDER BY COALESCE(c.updated_at, c.created_at) DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id,
                       COALESCE(
                         NULLIF(c.title, ''),
                         (
                           SELECT CASE
                             WHEN length(m.content) > 48 THEN substr(m.content, 1, 45) || '…'
                             ELSE m.content
                           END
                           FROM messages m
                           WHERE m.conversation_id = c.id AND m.role = 'user'
                           ORDER BY m.id ASC
                           LIMIT 1
                         ),
                         'Untitled chat'
                       ) AS title,
                       c.created_at,
                       c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                FROM conversations c
                WHERE c.user_id = ?
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
                ORDER BY COALESCE(c.updated_at, c.created_at) DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [_row(r) for r in rows]


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        if use_postgres():
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = %s",
                (conversation_id,),
            )
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, user_id),
            )
            return cur.rowcount > 0
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        return cur.rowcount > 0


def clear_conversation(conversation_id: str, user_id: str) -> None:
    if not user_owns_conversation(conversation_id, user_id):
        return
    with get_connection() as conn:
        conn.execute(
            q("DELETE FROM messages WHERE conversation_id = %s"),
            (conversation_id,),
        )


# ── Memories ───────────────────────────────────────────────────────


def add_memory(mem_id: str, text: str, category: str, confidence: int, user_id: str) -> None:
    with get_connection() as conn:
        if use_postgres():
            conn.execute(
                """
                INSERT INTO memories (id, user_id, text, category, confidence)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET text = EXCLUDED.text, category = EXCLUDED.category, confidence = EXCLUDED.confidence
                """,
                (mem_id, user_id, text, category, confidence),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (id, user_id, text, category, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (mem_id, user_id, text, category, confidence),
            )


def get_all_memories(user_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            q(
                """
                SELECT id, text, category, confidence FROM memories
                WHERE user_id = %s
                ORDER BY created_at DESC
                """
            ),
            (user_id,),
        ).fetchall()
    return [_row(r) for r in rows]


def delete_memory(mem_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            q("DELETE FROM memories WHERE id = %s AND user_id = %s"),
            (mem_id, user_id),
        )
        return cur.rowcount > 0


# ── Documents ──────────────────────────────────────────────────────


def save_document(
    doc_id: str,
    filename: str,
    page_count: int,
    chunk_count: int,
    preview: str,
    user_id: str,
    content_hash: str | None = None,
) -> None:
    with get_connection() as conn:
        if use_postgres():
            conn.execute(
                """
                INSERT INTO documents
                    (id, user_id, filename, page_count, chunk_count, extracted_preview, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    page_count = EXCLUDED.page_count,
                    chunk_count = EXCLUDED.chunk_count,
                    extracted_preview = EXCLUDED.extracted_preview,
                    content_hash = EXCLUDED.content_hash
                """,
                (doc_id, user_id, filename, page_count, chunk_count, preview[:2000], content_hash),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (id, user_id, filename, page_count, chunk_count, extracted_preview, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, user_id, filename, page_count, chunk_count, preview[:2000], content_hash),
            )


def list_documents(user_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            q(
                """
                SELECT id, filename, page_count, chunk_count, content_hash, created_at
                FROM documents
                WHERE user_id = %s
                ORDER BY created_at DESC
                """
            ),
            (user_id,),
        ).fetchall()
    return [_row(r) for r in rows]


def get_document(doc_id: str, user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            q(
                """
                SELECT id, filename, page_count, chunk_count, extracted_preview, content_hash, created_at
                FROM documents WHERE id = %s AND user_id = %s
                """
            ),
            (doc_id, user_id),
        ).fetchone()
    return _row(row)


def get_document_by_hash(content_hash: str, user_id: str) -> dict | None:
    if not content_hash:
        return None
    with get_connection() as conn:
        row = conn.execute(
            q(
                """
                SELECT id, filename, page_count, chunk_count, extracted_preview, content_hash, created_at
                FROM documents
                WHERE content_hash = %s AND user_id = %s
                ORDER BY created_at ASC
                LIMIT 1
                """
            ),
            (content_hash, user_id),
        ).fetchone()
    return _row(row)


def delete_document(doc_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            q("DELETE FROM documents WHERE id = %s AND user_id = %s"),
            (doc_id, user_id),
        )
        return cur.rowcount > 0
