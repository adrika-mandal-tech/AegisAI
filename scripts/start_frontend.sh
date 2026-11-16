#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
echo "Starting AegisAI Frontend (http://localhost:3001)"
npm run dev -- -p 3001
