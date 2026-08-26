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
- Batched Gemini embeddings (768-d, L2-normalized)
- **Neon Postgres + pgvector** in production (SQLite + Chroma locally)
- **Google Sign-In** + **Continue as guest** (with persistence disclaimer)
- Multi-document selection per chat (per-user isolation)
- SHA-256 upload deduplication **per user**
- Source citations (filename + page)
- User memory (rule-based, per user)
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

Create `.env` in the project root (see `.env.example`):

```env
GEMINI_API_KEY=AIza-your-key-here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
EMBED_BATCH_SIZE=16
EMBED_REQUEST_DELAY=0.7
COOKIE_SECURE=false
# Optional but recommended for durable multi-user data:
# DATABASE_URL=postgresql://...@...neon.tech/neondb?sslmode=require
# GOOGLE_CLIENT_ID=....apps.googleusercontent.com
```

Without `DATABASE_URL`, GenRAG uses **local SQLite + Chroma** (fine for solo testing).  
With Neon `DATABASE_URL`, it uses **Postgres + pgvector** (signed-in data survives Render restarts).

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — auth gate first, then the app.

### 2. Google Sign-In setup (free)

1. [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials  
2. Create **OAuth client ID** → application type **Web application**  
3. Authorized JavaScript origins: `http://localhost:8000` and your Render URL  
4. Authorized redirect URIs: same origins  
5. Put the client ID in `.env` as `GOOGLE_CLIENT_ID`  
6. On Render, add the same env var  

Guest mode works without Google. Google button appears once `GOOGLE_CLIENT_ID` is set.

### 3. Neon Postgres (recommended free DB)

1. Create a project at [neon.tech](https://neon.tech)  
2. Copy the connection string into `DATABASE_URL`  
3. Restart the API — tables + `vector` extension are created automatically  
4. Re-upload PDFs after switching from SQLite (vectors move to pgvector)

### 4. Try it

1. Continue as guest **or** Continue with Google  
2. Upload one or more PDFs  
3. Select documents and ask a question  
4. Check **Sources** for filename + page  

## API overview

| Endpoint | Purpose |
|----------|---------|
| `GET /auth/config` | Public Google client id flag |
| `GET /auth/me` | Current session user |
| `POST /auth/guest` | Start guest session (cookie) |
| `POST /auth/google` | Exchange Google ID token for session |
| `POST /auth/logout` | Clear session |
| `POST /documents/upload` | Upload PDF (user-scoped) |
| `GET /documents` | List **your** documents |
| `DELETE /documents/{id}` | Remove document + vectors |
| `POST /chat` | Chat (`document_ids` for multi-doc RAG) |
| `GET /conversations` | List **your** chats |
| `GET /conversations/{id}` | Load chat messages |
| `DELETE /conversations/{id}` | Delete a chat |
| `GET /memories` | List memories |
| `GET /debug/last` | Last retrieval snapshot |

## Project structure

```
backend/
  main.py          # API routes + static UI
  auth.py          # Google + guest sessions
  db.py            # Postgres / SQLite connection
  database.py      # Users, chats, docs, memories
  ingestion.py     # PDF → text
  chunking.py      # text → chunks
  embeddings.py    # batched Gemini embeddings (768-d)
  vector_store.py  # pgvector or Chroma
  memory.py        # user memory
  rag.py           # retrieval + prompt + modes
  llm.py           # Gemini chat
frontend/
  index.html       # UI + auth gate
  style.css
data/              # local sqlite/chroma only (gitignored)
```

## Deploy (free)

One Render Docker service serves UI + API.

**Stack:** Render (app) + Neon (Postgres/pgvector) + Google OAuth + Gemini free tier.

**Limits:** Render sleeps when idle (~1 min cold start). Neon free has compute/storage caps — enough for a portfolio/demo.

### Render

1. Push this repo to GitHub  
2. **New → Web Service** → Docker → Free  
3. Health check: `/health`  
4. Env vars:  
   - `GEMINI_API_KEY`  
   - `DATABASE_URL` (Neon)  
   - `GOOGLE_CLIENT_ID`  
   - `COOKIE_SECURE=true`  
5. Deploy — open the Render URL  

Guest disclaimer: guest data is temporary. Google accounts persist in Neon across deploys.

## Author

[Ahmad Tayyab](https://github.com/ahmedtayyab)
