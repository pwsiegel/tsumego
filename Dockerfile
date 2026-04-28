# Single-image deployment: builds the React frontend, installs the FastAPI
# backend, and bakes in the ONNX YOLO weights. Inference runs on
# onnxruntime — torch/ultralytics are training-only and stay out of the
# serving image.

# ---------- Stage 1: frontend ----------
FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


# ---------- Stage 2: backend dependencies ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS backend
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Base deps include onnxruntime + opencv-python-headless. The [ml] extras
# (torch, ultralytics, ...) are NOT installed — they're for training only.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project

COPY backend/src ./src
RUN uv pip install --no-deps -e .


# ---------- Stage 3: runtime ----------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GOAPP_FRONTEND_DIR=/app/frontend \
    GOAPP_DATA_DIR=/data \
    PORT=8080

COPY --from=backend /app/.venv /app/.venv
COPY --from=backend /app/src /app/src
COPY --from=frontend /web/dist /app/frontend
# Mirror the repo layout so paths.py's repo-relative MODELS_DIR resolves
# to /app/data/models (parents[2] from /app/src/goapp/paths.py). Only the
# .onnx weights ship — .pt files are training artifacts.
COPY backend/data/models/*.onnx /app/data/models/

EXPOSE 8080
CMD ["sh", "-c", "uvicorn goapp.api:app --host 0.0.0.0 --port ${PORT}"]
