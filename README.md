# GenRAG

A RAG-based chatbot that lets you upload PDFs and ask questions about them — with multi-document chat, persistent library, Learning/Interview modes, and source citations.

## What GenRAG is

GenRAG is a local **document learning workspace**:

- Upload PDFs into a persistent **document library**
- Select one or more documents for a chat
- Ask questions grounded in retrieved chunks
- See **sources** (filename + page)
- Keep separate **chat history** without re-uploading documents

## Current architecture

```
PDF
 → text extraction (pypdf)
 → chunking (1500 chars / 150 overlap)
 → batched embeddings (Gemini gemini-embedding-001)
 → ChromaDB (one collection per document)
 → retrieval (cosine similarity across selected docs)
 → Gemini chat (gemini-3.5-flash)
 → answer + sources
```

### Architecture after improvements

```
Document Library
      ↓
PDF Upload (+ SHA-256 dedup)
      ↓
Text Extraction
      ↓
Chunking
      ↓
Batched Embeddings
      ↓
ChromaDB
      ↓
User Question
      ↓
Query Embedding (once)
      ↓
Selected Documents
      ↓
Similarity Search (per collection)
      ↓
Combined / ranked chunks
      ↓
Conversation History + RAG Context
      ↓
Gemini
      ↓
Answer + Sources
```

## Improvements made

- **Batched embedding requests** — many chunks per API call instead of one call per chunk
- **Multi-document chat** — select multiple PDFs; one query embedding searches all selected collections
- **Content-hash deduplication** — uploading the exact same PDF reuses existing embeddings
- **Improved document library** — checkboxes, search, clear selection, selected-count
- **Improved sources** — filename + page in the Sources panel and message meta
- **Fewer Gemini calls for titles** — titles from the first message text (no extra chat API call)

## Challenges encountered

Practical issues found while building this:

1. **Gemini embedding rate limits** — free-tier RPM made large PDFs fail when every chunk was a separate request
2. **One-request-per-chunk embedding** — a 100-page PDF could mean 200–300 API calls and long waits
3. **Duplicate uploads** — same file uploaded twice previously re-embedded everything
4. **Multiple documents in one conversation** — needed multi-select + merged retrieval without merging collections
5. **Document-to-chunk relationships** — keep one Chroma collection per document with filename/page metadata
6. **Combining multi-collection results** — search each collection, then rank by similarity score
7. **Source/page citations** — preserve filename + page through retrieval into the UI
8. **History vs document context** — chat messages stay in SQLite; document vectors stay in Chroma; both are injected into the prompt separately

## Why these changes were made

To keep GenRAG usable on Gemini free tier, avoid wasted embedding quota, and make the product feel like a real study workspace where documents persist and chats can use several of them at once.

## Features

- PDF upload + text extraction
- Chunking with overlap
- Batched Gemini embeddings + Chroma cosine search
- Multi-document selection per chat
- SHA-256 upload deduplication
- Source citations (filename + page)
- User memory (rule-based)
- Modes: **Chat**, **Learning**, **Interview**
- Persistent chat history with titles
- Debug panel for retrieval transparency

## Quick start

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=AIza-your-key-here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
EMBED_BATCH_SIZE=16
EMBED_REQUEST_DELAY=0.7
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

Or open the UI from the API itself: http://localhost:8000

### 3. Try it

1. Upload one or more PDFs
2. Check the documents you want for this chat
3. Ask a question that spans the selected material
4. Check **Sources** for filename + page
5. Start a **New chat** — documents stay in the library (not re-embedded)

## API overview

| Endpoint | Purpose |
|----------|---------|
| `POST /documents/upload` | Upload PDF (or reuse if hash exists) |
| `GET /documents` | List library documents |
| `DELETE /documents/{id}` | Remove document + vectors |
| `POST /chat` | Chat (`document_ids` for multi-doc RAG) |
| `GET /conversations` | List saved chats |
| `GET /conversations/{id}` | Load chat messages |
| `DELETE /conversations/{id}` | Delete a chat |
| `GET /memories` | List memories |
| `GET /debug/last` | Last retrieval snapshot |

## Project structure

```
backend/
  main.py          # API routes
  ingestion.py     # PDF → text
  chunking.py      # text → chunks
  embeddings.py    # batched Gemini embeddings
  vector_store.py  # Chroma store + multi-doc search
  memory.py        # user memory
  rag.py           # retrieval + prompt + modes
  llm.py           # Gemini chat
  database.py      # SQLite (+ content_hash)
frontend/
  index.html
  style.css
data/              # genrag.db, chroma (gitignored)
```

## Deploy (free)

The app is one service: FastAPI serves the API **and** the frontend.

**Limits of the free tier:** the instance sleeps after idle time (first load can take ~1 minute), and uploaded PDFs/chats live on ephemeral disk — they reset when the service restarts.

### Render

1. Push this repo to GitHub
2. Open [Render](https://render.com) → **New** → **Web Service** → connect the repo
3. Settings:
   - **Runtime:** Docker
   - **Instance type:** Free
   - **Health check path:** `/health`
4. Add environment variable:
   - `GEMINI_API_KEY` = your key from [Google AI Studio](https://aistudio.google.com/apikey)
5. Deploy. Your public URL is the whole app (no separate frontend host)

Optional: **New** → **Blueprint** and select this repo (`render.yaml`).

### Local check before deploy

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — you should see the GenRAG UI.

## Author

[Ahmad Tayyab](https://github.com/ahmedtayyab)
