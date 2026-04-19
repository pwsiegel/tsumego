# Backend: FastAPI service for Go problem workbook.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync


FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src

EXPOSE 8001
CMD ["uvicorn", "goapp_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
