# RAG orchestration + Learning/Interview modes (Phases 7–12)

from dataclasses import asdict

from debug_state import DebugSnapshot, set_debug
from embeddings import embed_text
from llm import ask_llm_with_system
from memory import MemoryRecord, find_relevant_memories, try_extract_memory
from vector_store import SearchResult, search_documents


def build_rag_response(
    user_message: str,
    history: list[dict],
    mode: str,
    user_id: str,
    document_ids: list[str] | None = None,
    document_filenames: dict[str, str] | None = None,
) -> dict:
    extracted = try_extract_memory(user_message, user_id)
    memories = find_relevant_memories(user_message, user_id)

    selected_ids = [d for d in (document_ids or []) if d]
    chunks: list[SearchResult] = []
    if selected_ids:
        query_vector = embed_text(user_message)
        chunks = search_documents(
            selected_ids,
            query_vector,
            filenames=document_filenames or {},
            top_k_per_doc=3,
            final_top_k=6,
            user_id=user_id,
        )

    selected_names = [
        (document_filenames or {}).get(doc_id, doc_id) for doc_id in selected_ids
    ]
    system_prompt = _build_system_prompt(mode, memories, chunks, selected_names)
    final_user_message = _build_user_message(mode, user_message, chunks)

    reply = ask_llm_with_system(system_prompt, history, final_user_message)

    sources = _format_sources(chunks)
    memories_used = [asdict(m) for m in memories]

    set_debug(
        DebugSnapshot(
            mode=mode,
            document_id=",".join(selected_ids) if selected_ids else None,
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
        result["memory_saved"] = asdict(extracted)
    return result


def _build_system_prompt(
    mode: str,
    memories: list[MemoryRecord],
    chunks: list[SearchResult],
    selected_doc_names: list[str] | None = None,
) -> str:
    memory_block = "\n".join(f"- {m.text}" for m in memories) if memories else "(none)"
    selected = selected_doc_names or []

    if chunks:
        context_block = "\n\n---\n\n".join(
            f"[{i}] {c.filename or 'Document'} | Page {c.page}\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
    elif selected:
        names = ", ".join(selected)
        context_block = (
            f"(Documents selected: {names}. No matching chunks were retrieved for this question. "
            "Say you could not find relevant passages in the selected documents.)"
        )
    else:
        context_block = (
            "(no documents selected — answer from general knowledge, "
            "and tell the user to select documents in the sidebar for document-grounded answers)"
        )

    base = f"""You are GenRAG, a document learning assistant.

USER MEMORIES (personal facts about this learner — use for personalization):
{memory_block}

DOCUMENT CONTEXT (retrieved chunks — prefer these for factual answers):
{context_block}

Rules:
- Answer using DOCUMENT CONTEXT when the answer is there.
- If documents are selected but the context has no useful passages, say you could not find that in the selected documents.
- If no documents are selected, say so clearly and answer from general knowledge if appropriate.
- Do not invent page numbers, filenames, or quotes not present in context.
- Do not say "no document uploaded" if documents are listed as selected above.
- When using document context, cite sources inline with bracket numbers matching the context labels, e.g. [1] or [2]. Place citations at the end of the sentence they support.
- Prefer clear markdown: short headings, bullet lists, and concise paragraphs.
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
    return [
        {
            "filename": c.filename or "Document",
            "document_id": c.document_id,
            "page": c.page,
            "score": c.score,
            "preview": c.text[:120],
        }
        for c in chunks
    ]
