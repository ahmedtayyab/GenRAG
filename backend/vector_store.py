# Store embeddings + search by cosine similarity (Phases 5 & 6)

import json
import math
from dataclasses import asdict, dataclass
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


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "vectors"


def save_document_vectors(document_id: str, filename: str, chunks: list[StoredChunk]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{document_id}.json"
    payload = {
        "document_id": document_id,
        "filename": filename,
        "chunks": [asdict(c) for c in chunks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")  # educational store — production uses vector DB
    return path


def load_document_vectors(document_id: str) -> dict | None:
    path = DATA_DIR / f"{document_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_document_vectors(document_id: str) -> bool:
    path = DATA_DIR / f"{document_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def search(document_id: str, query_vector: list[float], top_k: int = 3) -> list[SearchResult]:
    doc = load_document_vectors(document_id)
    if not doc:
        return []

    scored: list[SearchResult] = []
    for chunk in doc["chunks"]:
        score = cosine_similarity(query_vector, chunk["vector"])  # compare question vector to each chunk vector
        scored.append(
            SearchResult(
                index=chunk["index"],
                text=chunk["text"],
                page=chunk["page"],
                score=round(score, 4),
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)  # highest similarity first
    return scored[:top_k]  # return top-k most relevant chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))  # dot product measures alignment of vectors
    norm_a = math.sqrt(sum(x * x for x in a))  # length of vector a
    norm_b = math.sqrt(sum(x * x for x in b))  # length of vector b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)  # 1.0 = identical direction, 0 = unrelated
