"""GenRAG API — Phase 1: basic chat with persisted conversation history."""

from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import add_message, clear_conversation, get_recent_history, init_db
from llm import ask_llm

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


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    phase: str
    product: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", phase="1", product="GenRAG")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid4())

    try:
        history = get_recent_history(conversation_id)
        reply = ask_llm(request.message, history)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    add_message(conversation_id, "user", request.message)
    add_message(conversation_id, "assistant", reply)

    return ChatResponse(reply=reply, conversation_id=conversation_id)


@app.post("/chat/reset")
def reset_chat(conversation_id: str):
    clear_conversation(conversation_id)
    return {"status": "ok", "conversation_id": conversation_id}


@app.get("/chat/history/{conversation_id}")
def get_history(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": get_recent_history(conversation_id, limit=100)}
