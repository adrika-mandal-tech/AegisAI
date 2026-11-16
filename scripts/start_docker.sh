#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../docker"
echo "🚀 Starting AegisAI infrastructure (Qdrant)..."
docker compose -f docker-compose.yml up -d qdrant
echo "✅ Qdrant running at http://localhost:6333"
