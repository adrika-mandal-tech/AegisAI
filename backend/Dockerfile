# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system runtime dependencies required for Git repo ingestion, OpenCV/OCR/OpenMP, and healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl libgomp1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy project definition, readme, and package source from build context (project root)
COPY backend/pyproject.toml .
COPY README.md .
COPY backend/aegis ./aegis
COPY backend/tests ./tests

# Fix readme path for build context layout inside the container
RUN sed -i 's|readme = "../README.md"|readme = "README.md"|' pyproject.toml

# Install dependencies into a virtual environment and ensure persistent data directory exists
RUN python -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -e ".[dev]" && \
    mkdir -p /app/data

EXPOSE 8000
CMD [".venv/bin/uvicorn", "aegis.main:app", "--host", "0.0.0.0", "--port", "8000"]
