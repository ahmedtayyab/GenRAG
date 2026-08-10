# GenRAG

**General document learning assistant** — upload documents, ask questions with RAG, study with memory, Learning Mode, and Interview Mode.

Built incrementally as a learning project to understand AI assistant infrastructure: conversation history, prompts, files, memory, embeddings, retrieval, and generation.

## Current status: Phase 1

Basic chatbot with:

- FastAPI backend
- OpenAI chat completions
- Conversation history in SQLite (last 10 turns sent to the LLM)
- Simple web UI (GenRAG theme)

Coming next: PDF upload (Phase 2), chunking, embeddings, vector storage, retrieval, citations, memory, modes.

## Quick start

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and set your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

Run the API:

```powershell
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs

### 2. Frontend

In a second terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open http://localhost:5500

### 3. Test

1. Send: `Say hello in exactly five words.`
2. Send: `My name is Ahmad.` then `What's my name?` — confirms history works.
3. Click **Reset conversation**, ask your name again — should not remember.

## Project structure

```
genrag/
├── backend/
│   ├── main.py       # FastAPI routes
│   ├── llm.py        # OpenAI client
│   ├── database.py   # SQLite (history + memory placeholder)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   └── style.css
├── design.md         # GenRAG UI design tokens
├── .env.example
└── README.md
```

## Design

See [design.md](design.md) for colors, typography, and layout guidelines.

## Author

[Ahmad Tayyab](https://github.com/ahmedtayyab)

## License

MIT
