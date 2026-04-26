# Single-image deployment: builds the React frontend, installs the FastAPI
# backend with CPU-only torch, and bakes in the YOLO model weights.

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

# Base FastAPI deps (no torch yet). Lockfile pinned for reproducibility.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project

# CPU-only torch from PyTorch's index — saves ~1 GB vs the default
# CUDA-enabled wheel. ultralytics picks up the already-installed torch.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision
RUN uv pip install ultralytics onnx onnxruntime

# ultralytics depends on opencv-python (GUI wheel), which clashes with the
# headless variant in our base deps. Force-reinstall headless so cv2's files
# end up from the headless wheel (works without libGL beyond what we apt-install).
RUN uv pip install --force-reinstall --no-deps "opencv-python-headless>=4.10"

# Install the project itself without touching deps (avoids resync that would
# evict the manually-installed ml packages above).
COPY backend/src ./src
RUN uv pip install --no-deps -e .


# ---------- Stage 3: runtime ----------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# libgl + libglib for opencv-python-headless's runtime needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GOAPP_FRONTEND_DIR=/app/frontend \
    GOAPP_DATA_DIR=/data \
    PORT=8080

COPY --from=backend /app/.venv /app/.venv
COPY --from=backend /app/src /app/src
COPY --from=frontend /web/dist /app/frontend
# Mirror the repo layout so paths.py's repo-relative MODELS_DIR resolves
# to /app/data/models (parents[2] from /app/src/goapp/paths.py).
COPY backend/data/models /app/data/models

EXPOSE 8080
CMD ["sh", "-c", "uvicorn goapp.api:app --host 0.0.0.0 --port ${PORT}"]
