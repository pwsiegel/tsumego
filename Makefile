.PHONY: help setup api web dev lint \
       synth train-boards train-stones train-grid validate \
       docker-up docker-down \
       deploy logs \
       modal-upload-synth modal-train-smoke modal-train-boards modal-train-stones modal-train-grid modal-pull-weights

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: ## Install all dependencies (backend + frontend)
	uv --directory backend sync --extra ml --extra dev
	cd web && npm install

# ---------------------------------------------------------------------------
# Dev servers
# ---------------------------------------------------------------------------

api: ## Start backend dev server (port 8001)
	uv --directory backend run uvicorn goapp.api:app --reload --port 8001

web: ## Start frontend dev server (port 5173)
	cd web && npm run dev

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

lint: ## Lint backend (ruff) + frontend (eslint)
	uv --directory backend run --extra dev ruff check src/
	cd web && npm run lint

# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------

DEVICE ?= mps

synth: ## Generate synthetic training pages (CPU, ~5 min)
	uv --directory backend run python -m goapp.synth.gen --count 1500

train-boards: ## Train the board detector
	uv --directory backend run --extra ml python -m goapp.ml.board_detect.train \
		--limit 500 --epochs 25

train-stones: ## Train the stone detector (set DEVICE=cpu if no GPU)
	uv --directory backend run --extra ml python -m goapp.ml.stone_detect.train \
		--limit 1500 --epochs 30 --device $(DEVICE)

train-grid: ## Train the grid-geometry regressor
	uv --directory backend run --extra ml python -m goapp.ml.grid_detect.train \
		--limit 1500 --epochs 40 --device $(DEVICE)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

DATASET ?= hm2
VAL_DIR ?= $(HOME)/data/go-app/data/val/$(DATASET)
MODEL   ?= backend/data/models/stone_detector.pt

validate: ## Run pipeline against val dataset and print report
	uv --directory backend run python -m goapp.cli.compare_on_val \
		--val-dir $(VAL_DIR) --model $(MODEL) --status accepted --verbose

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-up: ## Build and start both services via docker compose
	docker compose up --build

docker-down: ## Stop docker compose services
	docker compose down

# ---------------------------------------------------------------------------
# Cloud Run deploy
# ---------------------------------------------------------------------------

GCP_PROJECT ?= tsumego-pwsiegel
GCP_REGION  ?= us-central1
GCP_SERVICE ?= tsumego
GCP_IMAGE   := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/apps/$(GCP_SERVICE):latest

deploy: ## Build via Cloud Build and roll out to Cloud Run
	gcloud builds submit --tag $(GCP_IMAGE) --region=$(GCP_REGION) --project=$(GCP_PROJECT) .
	gcloud run deploy $(GCP_SERVICE) \
		--image=$(GCP_IMAGE) \
		--region=$(GCP_REGION) \
		--project=$(GCP_PROJECT)

logs: ## Tail recent Cloud Run logs (last 50 lines)
	gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="$(GCP_SERVICE)"' \
		--project=$(GCP_PROJECT) --limit=50 --format='value(timestamp,textPayload)' --freshness=10m

# ---------------------------------------------------------------------------
# Modal training (L4 GPUs on Modal)
# ---------------------------------------------------------------------------

MODAL_VOLUME    := tsumego-data
SYNTH_LOCAL_DIR := $(HOME)/data/go-app/data/synth_pages

modal-upload-synth: ## Upload local synth_pages/ to the Modal volume (one-time)
	modal volume create $(MODAL_VOLUME) 2>/dev/null || true
	modal volume put --force $(MODAL_VOLUME) $(SYNTH_LOCAL_DIR) /data/synth_pages

modal-train-smoke: ## Run a tiny ~2 min smoke test on Modal (L4)
	modal run training/modal_train.py::smoke

modal-train-boards: ## Train the board detector on Modal (L4)
	modal run training/modal_train.py::train_boards

modal-train-stones: ## Train the stone detector on Modal (L4)
	modal run training/modal_train.py::train_stones

modal-train-grid: ## Train the grid detector on Modal (L4)
	modal run training/modal_train.py::train_grid

modal-pull-weights: ## Copy trained weights from the Modal volume into backend/data/models/
	modal volume get --force $(MODAL_VOLUME) /models/board_detector.pt backend/data/models/
	modal volume get --force $(MODAL_VOLUME) /models/stone_detector.pt backend/data/models/
	modal volume get --force $(MODAL_VOLUME) /models/grid_detector.pt backend/data/models/
