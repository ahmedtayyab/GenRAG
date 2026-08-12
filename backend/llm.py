# LLM client — Gemini chat (OpenAI-compatible API)

import os

from openai import OpenAI

_client: OpenAI | None = None

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey"
            )
        base_url = os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def ask_llm_with_system(system_prompt: str, history: list[dict], user_message: str) -> str:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]  # dynamic system prompt from RAG pipeline
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def generate_conversation_title(first_message: str) -> str:
    """Short ChatGPT-style title from the first user message."""
    fallback = _fallback_title(first_message)
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a short chat title like ChatGPT does. "
                        "3 to 6 words. Title case. No quotes. No ending punctuation. "
                        "Capture the topic, not the full question."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ],
            temperature=0.3,
        )
        title = (response.choices[0].message.content or "").strip()
        title = title.strip("\"'`").rstrip(".!?:;")
        title = " ".join(title.split())
        if not title or len(title) > 60:
            return fallback
        return title
    except Exception:
        return fallback


def _fallback_title(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= 48:
        return cleaned
    return cleaned[:45].rstrip() + "…"
