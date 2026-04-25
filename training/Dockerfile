# Training image for Vertex AI Custom Jobs (CUDA, L4 GPU). Built via
# Cloud Build using training/cloudbuild.yaml. Reads synth pages from the
# GCS FUSE mount at /gcs/<bucket>/, writes trained weights back there.

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

# libgl + glib for opencv-python-headless's runtime needs (matches serving image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# All deps, including the ml extra (CUDA torch wheel from default PyPI index).
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project --extra ml

# ultralytics drags in opencv-python (GUI wheel); force-reinstall headless so
# cv2 imports come from the headless variant. Same trick as the serving image.
RUN uv pip install --force-reinstall --no-deps "opencv-python-headless>=4.10"

# goapp source, installed without re-resolving deps.
COPY backend/src ./src
RUN uv pip install --no-deps -e .

COPY training/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
