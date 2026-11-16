# AegisAI: AgentSwarm

Multi-agent document intelligence platform: a FastAPI + LangGraph backend running an
8-agent swarm (Supervisor, Planner, Retriever, Researcher, Validator, Reasoner, Critic,
Report Generator, plus an Evaluator), and a Next.js workspace frontend.

## Features

| Section | Endpoint | What it does |
| --- | --- | --- |
| Conversational chat | `POST /api/chat/stream` | Streams per-agent SSE events and a grounded answer with citations and charts |
| Agent execution | `GET /api/pipeline/stream` | Runs the full 8-agent pipeline and exports PDF, DOCX, PPTX, XLSX, CSV, HTML and Markdown |
| Documents | `/api/documents/*` | Upload files (up to 500 MB each), ingest URLs and Git repos, view chunks |
| Analytics lab | `POST /api/analytics/profile/{id}` | pandas profiling, outliers, correlations and charts for CSV/XLSX/JSON |
| Deliverables | `/api/artifacts/*` | Lists and downloads generated files per execution |

## Requirements

- Python 3.12+ (developed on 3.14)
- Node.js 18+
- A Gemini API key and/or a Groq API key. Tavily is optional and enables web research.
- About 3 GB of disk for Python dependencies. The `BAAI/bge-m3` embedding model
  (about 2.3 GB) downloads on first use.

Docker is **not** required. Vectors are stored in an embedded Qdrant database at
`backend/data/qdrant`.

## Quick start

```bash
./scripts/setup.sh                 # creates backend/.venv, installs deps, creates backend/.env
# edit backend/.env and set AEGIS_GEMINI_API_KEY and/or AEGIS_GROQ_API_KEY

./scripts/start_backend.sh         # terminal 1 -> http://localhost:8000/docs
./scripts/start_frontend.sh        # terminal 2 -> http://localhost:3001
```

Manual equivalent:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp ../.env.example .env            # then add your keys
.venv/bin/uvicorn aegis.main:app --host 0.0.0.0 --port 8000

cd ../frontend
npm install
npm run dev -- -p 3001
```

## Configuration notes

- `AEGIS_LLM_PROVIDER=auto` uses Gemini (`gemini-3.5-flash-lite`) first and switches to
  Groq (`openai/gpt-oss-120b`) on errors, timeouts or rate limits.
- Leave `AEGIS_QDRANT_URL` empty to use the embedded store. Only one backend process can
  open it at a time, so don't run uvicorn with `--reload` or with multiple workers.
- The frontend proxies `/api/*` to `AEGIS_BACKEND_URL` (see `frontend/.env.local`).

## Tests

```bash
cd backend && .venv/bin/pytest -m "not integration" -q     # 253 tests
cd frontend && npx tsc --noEmit
```

## Project layout

```
backend/aegis/agents/     the 8 agents + evaluator
backend/aegis/graph/      LangGraph state, nodes, edges, streaming
backend/aegis/api/        FastAPI routes and schemas
backend/aegis/llm/        Gemini / Groq / OpenLM clients + auto-fallback
backend/aegis/memory/     embeddings, Qdrant wrapper, retrieval
backend/aegis/services/   data analysis, visualization, artifact generation
backend/data/             documents, vector store, artifacts, registry (runtime data)
frontend/app, components  Next.js workspace UI
```
