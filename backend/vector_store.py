# Store embeddings + search via Chroma (Phases 5 & 6)

from dataclasses import dataclass
from pathlib import Path

import chromadb

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
    score: float  # cosine similarity 0-1, higher = more relevant
    document_id: str = ""
    filename: str = ""


CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _collection_name(document_id: str) -> str:
    return f"doc_{document_id.replace('-', '_')}"


def save_document_vectors(document_id: str, filename: str, chunks: list[StoredChunk]) -> Path:
    client = _get_client()
    name = _collection_name(document_id)

    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "document_id": document_id, "filename": filename},
    )

    collection.add(
        ids=[f"{document_id}_{c.index}" for c in chunks],
        embeddings=[c.vector for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"index": c.index, "page": c.page, "document_id": document_id, "filename": filename}
            for c in chunks
        ],
    )
    return CHROMA_DIR


def delete_document_vectors(document_id: str) -> bool:
    client = _get_client()
    name = _collection_name(document_id)
    try:
        client.delete_collection(name)
        return True
    except Exception:
        return False


def has_document_vectors(document_id: str) -> bool:
    """True if this document has a Chroma collection with at least one chunk."""
    client = _get_client()
    name = _collection_name(document_id)
    try:
        collection = client.get_collection(name)
        return collection.count() > 0
    except Exception:
        return False


def search(document_id: str, query_vector: list[float], top_k: int = 3, filename: str = "") -> list[SearchResult]:
    client = _get_client()
    name = _collection_name(document_id)
    try:
        collection = client.get_collection(name)
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
) -> list[SearchResult]:
    """Search each selected document collection, then merge and rank by score."""
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
            )
        )

    combined.sort(key=lambda r: r.score, reverse=True)
    return combined[:final_top_k]
