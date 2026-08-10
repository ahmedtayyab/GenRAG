# Store embeddings + search via Chroma (Phases 5 & 6)

import chromadb
from dataclasses import dataclass
from pathlib import Path

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


CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _collection_name(document_id: str) -> str:
    # Chroma names: 3–63 chars, alphanumeric + underscores/hyphens
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
        metadatas=[{"index": c.index, "page": c.page} for c in chunks],
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


def search(document_id: str, query_vector: list[float], top_k: int = 3) -> list[SearchResult]:
    client = _get_client()
    name = _collection_name(document_id)
    try:
        collection = client.get_collection(name)
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    scored: list[SearchResult] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        # cosine distance ≈ 1 - cosine_similarity for normalized vectors
        score = max(0.0, 1.0 - distance) if distance is not None else 0.0
        scored.append(
            SearchResult(
                index=meta["index"],
                text=results["documents"][0][i],
                page=meta["page"],
                score=round(score, 4),
            )
        )

    return scored
