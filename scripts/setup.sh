#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Setting up AegisAI..."

echo ""
echo "--- 1. Backend (Python 3.12+) ---"
PY="${PYTHON:-python3}"
if [ ! -d "backend/.venv" ]; then
  "$PY" -m venv backend/.venv
  echo "Created backend/.venv"
fi
backend/.venv/bin/pip install --upgrade pip
if ! (cd backend && .venv/bin/pip install -e ".[dev]"); then
  echo "Backend dependency installation failed."
  exit 1
fi
echo "Backend dependencies installed."

if [ ! -f "backend/.env" ]; then
  cp .env.example backend/.env
  echo "Created backend/.env from .env.example. Add your API keys before starting."
fi

echo ""
echo "--- 2. Frontend (Node 18+) ---"
if [ ! -f "frontend/.env.local" ]; then
  printf 'NEXT_PUBLIC_API_URL=/api\nAEGIS_BACKEND_URL=http://localhost:8000\n' > frontend/.env.local
  echo "Created frontend/.env.local"
fi
if ! (cd frontend && npm install); then
  echo "Frontend dependency installation failed."
  exit 1
fi
echo "Frontend dependencies installed."

echo ""
echo "Setup complete."
echo "  Backend:  ./scripts/start_backend.sh   -> http://localhost:8000/docs"
echo "  Frontend: ./scripts/start_frontend.sh  -> http://localhost:3001"
echo "  (Docker/Qdrant server is optional; the embedded store in backend/data/qdrant is used by default.)"
