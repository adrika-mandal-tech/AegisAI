#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
echo "Starting AegisAI Backend (http://localhost:8000, docs at /docs)"
# No --reload: the embedded Qdrant store allows only one process to open it.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/uvicorn aegis.main:app --host 0.0.0.0 --port 8000
