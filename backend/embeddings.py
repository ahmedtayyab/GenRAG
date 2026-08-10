# Text → embedding vectors (Phase 4) — Gemini via OpenAI-compatible API

import os
import time

from openai import OpenAI, RateLimitError

from llm import GEMINI_BASE_URL

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")  # free tier: ~100 embed requests/minute
REQUEST_DELAY = float(os.getenv("EMBED_REQUEST_DELAY", "0.7"))  # seconds between calls — keeps under rate limit
MAX_RETRIES = 5


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cleaned = [t.strip() for t in texts if t and t.strip()]  # skip empty chunks
    if not cleaned:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")

    client = OpenAI(api_key=api_key, base_url=os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL))
    vectors: list[list[float]] = []

    for i, text in enumerate(cleaned):
        vector = _embed_one_with_retry(client, text, i + 1, len(cleaned))
        vectors.append(vector)
        if i < len(cleaned) - 1:
            time.sleep(REQUEST_DELAY)  # free tier: max ~100 embed calls per minute

    return vectors


def _embed_one_with_retry(client: OpenAI, text: str, index: int, total: int) -> list[float]:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=text)
            return response.data[0].embedding
        except RateLimitError as exc:
            last_error = exc
            wait = 20 * (attempt + 1)  # 20s, 40s, 60s... when quota exceeded
            time.sleep(wait)
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                time.sleep(20 * (attempt + 1))
                continue
            raise ValueError(
                f"Embedding failed on chunk {index}/{total} using {EMBED_MODEL}: {exc}"
            ) from exc

    raise ValueError(
        f"Rate limit exceeded while embedding chunk {index}/{total}. "
        f"Free tier allows ~100 embeddings/minute. Your PDF has {total} chunks — "
        f"wait a minute and try again, or use a smaller PDF. Last error: {last_error}"
    ) from last_error
