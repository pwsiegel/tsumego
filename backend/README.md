# goapp backend

Backend service: PDF ingestion, stone detection, model training/inference.

## Setup

```bash
cd backend
uv sync                   # installs base deps
uv sync --extra ml        # when you need torch/onnx for training or inference
uv sync --extra dev       # tests + linter
```

## Run

```bash
uv run uvicorn goapp.api:app --reload --port 8001
```

## Test

```bash
uv run --extra dev pytest
```

## Layout

- `src/goapp/api/` — FastAPI route handlers, split by domain
- `src/goapp/ml/` — ML modules (board detection, stone detection, edge detection, pitch, discretization)
- `src/goapp/cli/` — CLI tools (validation runner, dataset export, model comparison)
- `src/goapp/synth/` — Synthetic data generation for training
