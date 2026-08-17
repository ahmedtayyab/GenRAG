# GenRAG API — full pipeline: upload → chunk → embed → retrieve → generate

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
load_dotenv(ROOT_DIR / ".env")

from chunking import chunk_pages
from database import (
    add_message,
    clear_conversation,
    count_messages,
    delete_conversation,
    delete_document,
    get_conversation,
    get_document,
    get_document_by_hash,
    get_full_history,
    get_recent_history,
    init_db,
    list_conversations,
    list_documents,
    save_document,
    set_conversation_title,
)
from debug_state import get_debug
from embeddings import embed_texts
from ingestion import extract_pdf_text
from llm import generate_conversation_title
from memory import list_memories, remove_memory
from rag import build_rag_response
from vector_store import StoredChunk, delete_document_vectors, has_document_vectors, save_document_vectors


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="GenRAG", description="General document learning assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    document_id: str | None = None  # backward compatible single selection
    document_ids: list[str] | None = None  # multi-document RAG
    mode: str = "chat"  # chat | learning | interview


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    title: str | None = None
    sources: list = []
    retrieved_chunks: list = []
    memories_used: list = []
    memory_saved: dict | None = None


def _resolve_document_ids(request: ChatRequest) -> list[str]:
    ids: list[str] = []
    if request.document_ids:
        ids.extend(request.document_ids)
    if request.document_id:
        ids.append(request.document_id)
    # preserve order, drop empties/duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


@app.get("/health")
def health():
    return {"status": "ok", "phase": "complete", "product": "GenRAG"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in this version.")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20 MB).")

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = get_document_by_hash(content_hash)
    if existing:
        return {
            "document_id": existing["id"],
            "filename": existing["filename"],
            "page_count": existing["page_count"],
            "chunk_count": existing["chunk_count"],
            "preview": existing.get("extracted_preview") or "",
            "already_exists": True,
            "message": "This exact file was already uploaded. Reusing existing embeddings.",
        }

    try:
        document_id = str(uuid4())
        pages = extract_pdf_text(file_bytes)
        if not pages:
            raise HTTPException(status_code=400, detail="No text found in PDF. Scanned PDFs need OCR.")

        chunks = chunk_pages(pages)
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)

        stored = [
            StoredChunk(index=c.index, text=c.text, page=c.page, vector=vectors[i])
            for i, c in enumerate(chunks)
        ]
        save_document_vectors(document_id, file.filename, stored)
        preview = "\n\n".join(p.text[:200] for p in pages[:3])
        save_document(
            document_id,
            file.filename,
            len(pages),
            len(chunks),
            preview,
            content_hash=content_hash,
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "preview": preview,
            "already_exists": False,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload processing failed: {exc}") from exc


@app.get("/documents")
def get_documents():
    docs = list_documents()
    for doc in docs:
        doc["has_vectors"] = has_document_vectors(doc["id"])
    return {"documents": docs}


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    if not get_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_document_vectors(document_id)
    delete_document(document_id)
    return {"status": "ok", "document_id": document_id}


@app.get("/memories")
def get_memories():
    return {"memories": [m.__dict__ for m in list_memories()]}


@app.delete("/memories/{memory_id}")
def delete_memory_route(memory_id: str):
    if not remove_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "ok"}


@app.get("/debug/last")
def debug_last():
    return get_debug()


@app.get("/conversations")
def get_conversations():
    return {"conversations": list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation_detail(conversation_id: str):
    convo = get_conversation(conversation_id)
    if not convo or count_messages(conversation_id) == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        **convo,
        "messages": get_full_history(conversation_id),
    }


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    if not delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "ok", "conversation_id": conversation_id}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid4())
    mode = request.mode if request.mode in ("chat", "learning", "interview") else "chat"
    document_ids = _resolve_document_ids(request)

    filenames: dict[str, str] = {}
    ready_ids: list[str] = []
    missing_vector_names: list[str] = []
    for doc_id in document_ids:
        doc = get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Selected document not found: {doc_id}. Upload a PDF first.",
            )
        filenames[doc_id] = doc["filename"]
        if has_document_vectors(doc_id):
            ready_ids.append(doc_id)
        else:
            missing_vector_names.append(doc["filename"])

    if document_ids and not ready_ids:
        names = ", ".join(missing_vector_names) or "selected documents"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Selected document(s) have no searchable embeddings: {names}. "
                "Delete them from Documents and re-upload the PDF(s) so they can be embedded again."
            ),
        )

    try:
        history = get_recent_history(conversation_id)
        result = build_rag_response(
            user_message=request.message,
            history=history,
            mode=mode,
            document_ids=ready_ids,
            document_filenames=filenames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    if missing_vector_names:
        skipped = ", ".join(missing_vector_names)
        result["reply"] = (
            f"(Note: skipped documents without embeddings: {skipped}. "
            "Delete and re-upload them to use them in chat.)\n\n"
            + result["reply"]
        )

    is_first_turn = count_messages(conversation_id) == 0
    add_message(conversation_id, "user", request.message)
    add_message(conversation_id, "assistant", result["reply"])

    title = None
    convo = get_conversation(conversation_id)
    if is_first_turn or not (convo and convo.get("title")):
        title = generate_conversation_title(request.message)
        set_conversation_title(conversation_id, title)
    else:
        title = convo.get("title")

    return ChatResponse(
        reply=result["reply"],
        conversation_id=conversation_id,
        title=title,
        sources=result.get("sources", []),
        retrieved_chunks=result.get("retrieved_chunks", []),
        memories_used=result.get("memories_used", []),
        memory_saved=result.get("memory_saved"),
    )


@app.post("/chat/reset")
def reset_chat(conversation_id: str):
    clear_conversation(conversation_id)
    return {"status": "ok", "conversation_id": conversation_id}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
