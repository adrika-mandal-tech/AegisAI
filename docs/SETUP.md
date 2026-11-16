# AegisAI — Setup Guide

## Prerequisites

- Python 3.12+ (tested on 3.14)
- Docker Desktop
- Node.js v22+
- [Google Gemini API key](https://aistudio.google.com/apikey)
- [Tavily API key](https://tavily.com)
- [LangSmith API key](https://smith.langchain.com) (optional)

## Quick Start

```bash
# 1. Start infrastructure (Qdrant vector database)
./scripts/start_docker.sh

# 2. Configure environment
# Edit backend/.env and set AEGIS_GEMINI_API_KEY
cp .env.example backend/.env

# 3. Install all dependencies (backend + frontend)
./scripts/setup.sh

# 4. Run tests (optional)
cd backend && .venv/bin/python -m pytest tests/ -v && cd ..

# 5. Start development servers (backend + frontend)
# This will run two processes in parallel in your terminal
./scripts/start_backend.sh & ./scripts/start_frontend.sh
```

## Production Start (Unified)

To build and run the entire application stack (Qdrant, backend, frontend) in production mode:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The application will be available at `http://localhost:3000`.
The Compose configuration routes browser API calls through Next.js to the
backend service, and routes backend Qdrant traffic to `qdrant:6333`. The
frontend image excludes local `.env` files from its build context.

The default embedding configuration is local BGE-M3 (`BAAI/bge-m3`, 1024
dimensions). Override `AEGIS_EMBEDDING_PROVIDER`, `AEGIS_EMBEDDING_MODEL`, and
`AEGIS_EMBEDDING_DIMENSIONS` together if using a different embedding provider;
existing Qdrant collections are not resized automatically.

## Key Design Decisions

1. **LangGraph for orchestration** — Existing Aegis agents run in a sequential workflow.
2. **Provider adapters** — Gemini and Groq use the configured provider/fallback settings.
3. **Qdrant for vector storage** — The default Compose deployment uses the Qdrant service.
4. **`pyproject.toml` is the Python dependency source** — No separate `requirements.txt`.
