# RAG orchestration + Learning/Interview modes (Phases 7–12)

from dataclasses import asdict

from debug_state import DebugSnapshot, set_debug
from embeddings import embed_text
from llm import ask_llm_with_system
from memory import MemoryRecord, find_relevant_memories, try_extract_memory
from vector_store import SearchResult, search


def build_rag_response(
    user_message: str,
    history: list[dict],
    mode: str,
    document_id: str | None,
) -> dict:
    extracted = try_extract_memory(user_message)  # check if user said "remember that..."
    memories = find_relevant_memories(user_message)  # pull relevant user facts into prompt

    chunks: list[SearchResult] = []
    if document_id:
        query_vector = embed_text(user_message)  # turn question into numbers for similarity search
        chunks = search(document_id, query_vector, top_k=3)  # find top 3 matching document chunks

    system_prompt = _build_system_prompt(mode, memories, chunks)  # assemble instructions + context
    final_user_message = _build_user_message(mode, user_message, chunks)  # mode-specific user message wrapper

    reply = ask_llm_with_system(system_prompt, history, final_user_message)  # send everything to Gemini

    sources = _format_sources(chunks)  # page citations for UI
    memories_used = [asdict(m) for m in memories]

    set_debug(
        DebugSnapshot(
            mode=mode,
            document_id=document_id,
            question=user_message,
            memories_used=memories_used,
            retrieved_chunks=[asdict(c) for c in chunks],
            history_count=len(history),
            system_prompt=system_prompt,
            final_user_message=final_user_message,
        )
    )

    result = {
        "reply": reply,
        "sources": sources,
        "retrieved_chunks": [asdict(c) for c in chunks],
        "memories_used": memories_used,
    }
    if extracted:
        result["memory_saved"] = asdict(extracted)  # tell UI a new memory was stored
    return result


def _build_system_prompt(mode: str, memories: list[MemoryRecord], chunks: list[SearchResult]) -> str:
    memory_block = "\n".join(f"- {m.text}" for m in memories) if memories else "(none)"
    context_block = "\n\n---\n\n".join(
        f"[Source: Page {c.page} | score {c.score}]\n{c.text}" for c in chunks
    ) if chunks else "(no document loaded — answer from general knowledge or say you need a document)"

    base = f"""You are GenRAG, a document learning assistant.

USER MEMORIES (personal facts about this learner — use for personalization):
{memory_block}

DOCUMENT CONTEXT (retrieved chunks — prefer these for factual answers):
{context_block}

Rules:
- Answer using DOCUMENT CONTEXT when the answer is there.
- If the answer is NOT in the document context, say clearly that it is not in the uploaded document.
- Do not invent page numbers or quotes not present in context.
- Cite pages when using document context (e.g. "Pages 3–4").
"""

    if mode == "learning":
        base += """
LEARNING MODE — structure every answer as:

### Simple explanation
### How it works
### Example
### Why it matters
### Important distinction
### Check your understanding
(End with one short question to test the learner.)
"""
    elif mode == "interview":
        base += """
INTERVIEW MODE — you simulate a technical researcher interviewing the learner about AI assistant infrastructure.

After the learner's answer, respond with:

**What you got right:**
**What to improve:**
**Concepts you missed:**
**Follow-up question:**

Be conversational, not multiple-choice. If the answer is weak, teach briefly then ask an easier follow-up.
If strong, ask a deeper architecture/tradeoff question. Focus on understanding, not scoring.
"""
    return base


def _build_user_message(mode: str, user_message: str, chunks: list[SearchResult]) -> str:
    if mode == "interview" and not chunks:
        return f"Learner answer or response:\n{user_message}\n\nAsk the next interview question about AI assistant infrastructure."
    return user_message


def _format_sources(chunks: list[SearchResult]) -> list[dict]:
    if not chunks:
        return []
    pages = sorted({c.page for c in chunks})
    return [{"pages": pages, "chunks": [{"page": c.page, "score": c.score, "preview": c.text[:120]} for c in chunks]}]
