"""LLM client — Phase 1: basic chat completions."""

import os

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def ask_llm(user_message: str, history: list[dict] | None = None) -> str:
    """
    Send messages to the LLM and return the assistant reply.

    The API is stateless — history must be passed in every request by the caller.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are GenRAG, a helpful document learning assistant. "
                "Be clear and concise. Phase 1: general chat only — "
                "document Q&A arrives in later phases."
            ),
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""
