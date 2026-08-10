# GenRAG

**General document learning assistant** — upload PDFs, ask questions with RAG, study with memory, Learning Mode, and Interview Mode.

## Features (complete)

- PDF upload and text extraction
- Chunking with overlap
- Gemini embeddings + Chroma vector search (cosine similarity)
- ChromaDB for persistent local vector storage
- Source citations (page numbers)
- User memory (rule-based extract + keyword retrieval)
- Modes: **Chat**, **Learning**, **Interview**
- Debug panel showing retrieved chunks and scores

## Quick start

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in project root:

```env
GEMINI_API_KEY=AIza-your-key-here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBED_MODEL=text-embedding-004
```

```powershell
uvicorn main:app --reload
```

### 2. Frontend

```powershell
cd frontend
python -m http.server 5500
```

Open http://localhost:5500

### 3. Try it

1. Upload your study PDF (e.g. AI Assistant Infrastructure)
2. Click the document in the sidebar to select it
3. Ask: "How does conversation history work?"
4. Switch to **Learning** or **Interview** mode
5. Say: "Remember that my interview is on Friday"
6. Check the **Debug** panel for retrieved chunks and scores

## API overview

| Endpoint | Purpose |
|----------|---------|
| `POST /documents/upload` | Upload PDF → chunk → embed → store |
| `GET /documents` | List uploaded documents |
| `DELETE /documents/{id}` | Remove document and vectors |
| `POST /chat` | RAG chat (mode: chat, learning, interview) |
| `POST /chat/reset` | Clear conversation history |
| `GET /memories` | List user memories |
| `DELETE /memories/{id}` | Remove a memory |
| `GET /debug/last` | Last retrieval pipeline snapshot |

## Project structure

```
backend/
  main.py          # API routes
  ingestion.py     # PDF → text
  chunking.py      # text → chunks
  embeddings.py    # text → vectors (Gemini)
  vector_store.py  # Chroma store + similarity search
  memory.py        # user memory extract/retrieve
  rag.py           # prompt construction + modes
  llm.py           # Gemini chat
  database.py      # SQLite
frontend/
  index.html       # UI
  style.css
data/              # genrag.db, chroma (local, gitignored)
```

## Author

[Ahmad Tayyab](https://github.com/ahmedtayyab)
