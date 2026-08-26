# Vector store — pgvector on Neon (Postgres) or Chroma (local SQLite fallback)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from database import EMBED_DIM
from db import get_connection, q, use_postgres


@dataclass
class StoredChunk:
    index: int
    text: str
    page: int
    vector: list[float]


@dataclass
class SearchResult:
    index: int
    text: str
    page: int
    score: float
    document_id: str = ""
    filename: str = ""


CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
_client = None


def _chroma():
    global _client
    if _client is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _collection_name(document_id: str) -> str:
    return f"doc_{document_id.replace('-', '_')}"


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


def save_document_vectors(
    document_id: str,
    filename: str,
    chunks: list[StoredChunk],
    user_id: str,
) -> None:
    if use_postgres():
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM document_chunks WHERE document_id = %s AND user_id = %s",
                (document_id, user_id),
            )
            for c in chunks:
                if len(c.vector) != EMBED_DIM:
                    raise ValueError(
                        f"Embedding dim {len(c.vector)} != expected {EMBED_DIM}"
                    )
                conn.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, user_id, chunk_index, page, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        document_id,
                        user_id,
                        c.index,
                        c.page,
                        c.text,
                        _vec_literal(c.vector),
                    ),
                )
        return

    client = _chroma()
    name = _collection_name(document_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
        },
    )
    collection.add(
        ids=[f"{document_id}_{c.index}" for c in chunks],
        embeddings=[c.vector for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "index": c.index,
                "page": c.page,
                "document_id": document_id,
                "filename": filename,
                "user_id": user_id,
            }
            for c in chunks
        ],
    )


def delete_document_vectors(document_id: str, user_id: str | None = None) -> bool:
    if use_postgres():
        with get_connection() as conn:
            if user_id:
                cur = conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s AND user_id = %s",
                    (document_id, user_id),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s",
                    (document_id,),
                )
            return cur.rowcount >= 0
    client = _chroma()
    try:
        client.delete_collection(_collection_name(document_id))
        return True
    except Exception:
        return False


def has_document_vectors(document_id: str, user_id: str | None = None) -> bool:
    if use_postgres():
        with get_connection() as conn:
            if user_id:
                row = conn.execute(
                    """
                    SELECT 1 AS ok FROM document_chunks
                    WHERE document_id = %s AND user_id = %s
                    LIMIT 1
                    """,
                    (document_id, user_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT 1 AS ok FROM document_chunks
                    WHERE document_id = %s
                    LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
        return row is not None

    client = _chroma()
    try:
        return client.get_collection(_collection_name(document_id)).count() > 0
    except Exception:
        return False


def search(
    document_id: str,
    query_vector: list[float],
    top_k: int = 3,
    filename: str = "",
    user_id: str | None = None,
) -> list[SearchResult]:
    if use_postgres():
        with get_connection() as conn:
            sql = """
                SELECT c.chunk_index, c.page, c.content, d.filename,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = %s
            """
            params: list = [_vec_literal(query_vector), document_id]
            if user_id:
                sql += " AND c.user_id = %s"
                params.append(user_id)
            sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.extend([_vec_literal(query_vector), top_k])
            rows = conn.execute(sql, tuple(params)).fetchall()
        out: list[SearchResult] = []
        for row in rows:
            out.append(
                SearchResult(
                    index=row["chunk_index"],
                    text=row["content"],
                    page=row["page"],
                    score=round(float(row["score"] or 0), 4),
                    document_id=document_id,
                    filename=row["filename"] or filename or "Document",
                )
            )
        return out

    client = _chroma()
    try:
        collection = client.get_collection(_collection_name(document_id))
    except Exception:
        return []
    collection_meta = collection.metadata or {}
    fallback_name = filename or collection_meta.get("filename") or "Document"
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []
    scored: list[SearchResult] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] or {}
        distance = results["distances"][0][i]
        score = max(0.0, 1.0 - distance) if distance is not None else 0.0
        scored.append(
            SearchResult(
                index=meta.get("index", i),
                text=results["documents"][0][i],
                page=meta.get("page", 1),
                score=round(score, 4),
                document_id=meta.get("document_id") or document_id,
                filename=meta.get("filename") or fallback_name,
            )
        )
    return scored


def search_documents(
    document_ids: list[str],
    query_vector: list[float],
    filenames: dict[str, str] | None = None,
    top_k_per_doc: int = 3,
    final_top_k: int = 6,
    user_id: str | None = None,
) -> list[SearchResult]:
    filenames = filenames or {}
    combined: list[SearchResult] = []
    for doc_id in document_ids:
        if not doc_id:
            continue
        combined.extend(
            search(
                doc_id,
                query_vector,
                top_k=top_k_per_doc,
                filename=filenames.get(doc_id, ""),
                user_id=user_id,
            )
        )
    combined.sort(key=lambda r: r.score, reverse=True)
    return combined[:final_top_k]
