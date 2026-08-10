# User memory: extract → store → retrieve (Phase 9) — rule-based, like About Us

import re
import uuid
from dataclasses import dataclass

from database import add_memory, delete_memory, get_all_memories


@dataclass
class MemoryRecord:
    id: str
    text: str
    category: str
    confidence: int


REMEMBER_PATTERNS = [
    (r"remember that (.+)", "explicit"),
    (r"remember (.+)", "explicit"),
    (r"don't forget (.+)", "explicit"),
    (r"my interview is (.+)", "goal"),
    (r"i struggle with (.+)", "weak_topic"),
    (r"i'm weak on (.+)", "weak_topic"),
    (r"i prefer (.+)", "preference"),
]


def try_extract_memory(message: str) -> MemoryRecord | None:
    lowered = message.strip()
    for pattern, category in REMEMBER_PATTERNS:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            text = match.group(1).strip().rstrip(".")
            if len(text) < 3:
                return None
            mem_id = str(uuid.uuid4())[:8]
            add_memory(mem_id, text, category, 85)  # save to SQLite
            return MemoryRecord(id=mem_id, text=text, category=category, confidence=85)
    return None


def find_relevant_memories(query: str, top_k: int = 3) -> list[MemoryRecord]:
    memories = get_all_memories()
    if not memories:
        return []

    query_words = set(_tokenize(query))
    scored: list[tuple[float, dict]] = []

    for mem in memories:
        mem_words = set(_tokenize(mem["text"]))
        overlap = len(query_words & mem_words)  # keyword overlap — simple but predictable (About Us style)
        if any(w in query.lower() for w in ("remember", "interview", "struggle", "weak")):
            overlap += 0.5  # small boost when user seems to ask about personal context
        if overlap > 0:
            scored.append((overlap, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, mem in scored[:top_k]:
        results.append(
            MemoryRecord(
                id=mem["id"],
                text=mem["text"],
                category=mem["category"],
                confidence=mem["confidence"],
            )
        )
    return results


def list_memories() -> list[MemoryRecord]:
    return [
        MemoryRecord(id=m["id"], text=m["text"], category=m["category"], confidence=m["confidence"])
        for m in get_all_memories()
    ]


def remove_memory(memory_id: str) -> bool:
    return delete_memory(memory_id)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]
