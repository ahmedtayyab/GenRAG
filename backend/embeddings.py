# Text → embedding vectors (Phase 4) — Gemini via OpenAI-compatible API

import os
import time

from openai import OpenAI, RateLimitError

from llm import GEMINI_BASE_URL

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
REQUEST_DELAY = float(os.getenv("EMBED_REQUEST_DELAY", "0.7"))  # pause between batches
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))  # chunks per embedding API call
MAX_RETRIES = 5


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts using batched API calls; return vectors in the same order."""
    if not texts:
        return []

    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")

    client = OpenAI(api_key=api_key, base_url=os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL))
    vectors: list[list[float]] = []
    total = len(cleaned)
    batch_size = max(1, BATCH_SIZE)

    for start in range(0, total, batch_size):
        batch = cleaned[start : start + batch_size]
        batch_num = start // batch_size + 1
        batch_total = (total + batch_size - 1) // batch_size
        vectors.extend(_embed_batch_with_retry(client, batch, batch_num, batch_total, total))
        if start + batch_size < total:
            time.sleep(REQUEST_DELAY)

    return vectors


def _embed_batch_with_retry(
    client: OpenAI,
    batch: list[str],
    batch_num: int,
    batch_total: int,
    chunk_total: int,
) -> list[list[float]]:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            if len(response.data) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, got {len(response.data)}"
                )
            # Prefer index mapping when the API provides indexes; otherwise keep response order
            if all(getattr(item, "index", None) is not None for item in response.data):
                by_index = {item.index: item.embedding for item in response.data}
                return [by_index[i] for i in range(len(batch))]
            return [item.embedding for item in response.data]
        except RateLimitError as exc:
            last_error = exc
            time.sleep(20 * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                time.sleep(20 * (attempt + 1))
                continue
            raise ValueError(
                f"Embedding failed on batch {batch_num}/{batch_total} "
                f"({len(batch)} chunks) using {EMBED_MODEL}: {exc}"
            ) from exc

    raise ValueError(
        f"Rate limit exceeded while embedding batch {batch_num}/{batch_total}. "
        f"Free tier allows limited embedding requests/minute. Your PDF has {chunk_total} chunks — "
        f"wait a minute and try again. Last error: {last_error}"
    ) from last_error
