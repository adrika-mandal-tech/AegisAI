# AegisAI Architecture

This document provides a high-level overview of the AegisAI backend and frontend architecture.

## Backend Architecture

The backend is a FastAPI application built with Python 3.12+. It follows a clean architecture pattern with a clear separation of concerns.

```
backend/
└── aegis/                # Main Python package
    ├── agents/           # 9 standalone agents
    ├── api/              # FastAPI endpoints
    ├── core/             # Configuration, logging, constants
    ├── evaluation/       # LangSmith + metrics
    ├── graph/            # LangGraph pipeline
    ├── llm/              # Gemini 3.6 Flash client
    ├── memory/           # Qdrant + embeddings
    ├── prompts/          # System prompts for agents
    ├── schemas/          # Shared Pydantic contracts
    └── tools/            # External service wrappers
```

**Key Principles:**

- **Modular:** Each component is self-contained.
- **Standalone Agents:** Every agent can be tested independently via its own API endpoint.
- **LangGraph Orchestration:** A `StateGraph` defines the RAG pipeline.
- **`google-genai` v1 SDK:** Uses the modern Google AI SDK.
- **`pyproject.toml`:** Single source of truth for dependencies.
- **Dry-Run Mode:** The server starts and runs gracefully without API keys.

## Frontend Architecture

The frontend is a Next.js 14 single-page application (SPA).

```
frontend/
├── app/                  # Main page + layout
├── components/           # Reusable React components
└── lib/                  # API client to talk to backend
```

**Key Principles:**

- **SPA:** A single page with 3 tabs (Pipeline, Documents, History).
- **Server Components:** Uses Next.js App Router.
- **Tailwind + Headless UI:** For a modern and responsive design.

## Unified Infrastructure (Docker Compose)

The entire application (Qdrant, backend, frontend) can be started with a single command:

```bash
docker compose -f docker/docker-compose.yml up
```

The Docker Compose file defines three services: `qdrant`, `backend`, and `frontend`, with proper dependencies and health checks.
