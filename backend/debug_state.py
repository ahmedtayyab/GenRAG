# Last request debug snapshot — so the UI can show the RAG pipeline (Phase 7+)

from dataclasses import asdict, dataclass, field


@dataclass
class DebugSnapshot:
    mode: str = "chat"
    document_id: str | None = None
    question: str = ""
    memories_used: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    history_count: int = 0
    system_prompt: str = ""
    final_user_message: str = ""


_last_debug = DebugSnapshot()


def set_debug(snapshot: DebugSnapshot) -> None:
    global _last_debug
    _last_debug = snapshot


def get_debug() -> dict:
    return asdict(_last_debug)
