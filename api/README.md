# goapp-api

Backend service: PDF ingestion, stone detection, model training/inference.

## Setup

```bash
cd api
uv sync                   # installs base deps
uv sync --extra ml        # when you need torch/onnx for training or inference
uv sync --extra dev       # tests + linter
```

## Run

```bash
uv run uvicorn goapp_api.main:app --reload --port 8001
```

## Test

```bash
uv run --extra dev pytest
```

## Layout

- `src/goapp_api/main.py` — FastAPI entrypoint
- `src/goapp_api/schemas.py` — Pydantic request/response models
- `models/` — trained ONNX weights (not committed; see top-level README)
