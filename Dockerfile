# PhotoMind — multi-stage Docker build
# Stage 1: Frontend (React + Vite)
# Stage 2: Backend  (FastAPI + CrewAI + RL)

# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────────────────
FROM python:3.11-slim AS backend
WORKDIR /app

# System deps for torch, Pillow, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY eval/ eval/
COPY knowledge_base/ knowledge_base/
COPY viz/ viz/
COPY .env.example .env.example

# Copy built frontend from stage 1
COPY --from=frontend /app/web/dist web/dist

EXPOSE 8000

# On container start: download photos + photo_index.json from R2 (if configured),
# then launch the FastAPI server. $PORT is honored for Render/Fly/Cloud Run.
CMD ["sh", "-c", "python scripts/fetch_assets.py && uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
