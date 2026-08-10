# GenRAG API — full pipeline: upload → chunk → embed → retrieve → generate

from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chunking import chunk_pages
from database import (
    add_message,
    clear_conversation,
    delete_document,
    get_document,
    get_recent_history,
    init_db,
    list_documents,
    save_document,
)
from debug_state import get_debug
from embeddings import embed_texts
from ingestion import extract_pdf_text
from memory import list_memories, remove_memory
from rag import build_rag_response
from vector_store import StoredChunk, delete_document_vectors, save_document_vectors

load_dotenv()


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
    document_id: str | None = None  # which uploaded PDF to search (RAG)
    mode: str = "chat"  # chat | learning | interview


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    sources: list = []
    retrieved_chunks: list = []
    memories_used: list = []
    memory_saved: dict | None = None


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
        save_document(document_id, file.filename, len(pages), len(chunks), preview)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "preview": preview,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload processing failed: {exc}") from exc


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


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
    return get_debug()  # shows last retrieval, prompt, scores — educational transparency


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid4())
    mode = request.mode if request.mode in ("chat", "learning", "interview") else "chat"

    if request.document_id and not get_document(request.document_id):
        raise HTTPException(status_code=404, detail="Selected document not found. Upload a PDF first.")

    try:
        history = get_recent_history(conversation_id)
        result = build_rag_response(  # Phases 6–12: retrieve + prompt + generate
            user_message=request.message,
            history=history,
            mode=mode,
            document_id=request.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    add_message(conversation_id, "user", request.message)
    add_message(conversation_id, "assistant", result["reply"])

    return ChatResponse(
        reply=result["reply"],
        conversation_id=conversation_id,
        sources=result.get("sources", []),
        retrieved_chunks=result.get("retrieved_chunks", []),
        memories_used=result.get("memories_used", []),
        memory_saved=result.get("memory_saved"),
    )


@app.post("/chat/reset")
def reset_chat(conversation_id: str):
    clear_conversation(conversation_id)
    return {"status": "ok", "conversation_id": conversation_id}
