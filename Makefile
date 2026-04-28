.PHONY: help setup api web dev lint \
       synth train-boards train-stones validate export-models \
       ingest-pdf \
       docker-up docker-down \
       deploy logs \
       modal-upload-synth modal-gen-synth modal-train-smoke modal-train-boards modal-train-stones modal-pull-weights

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

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

DATASET ?= hm2
VAL_DIR ?= $(HOME)/data/go-app/data/val/$(DATASET)
MODEL   ?= $(CURDIR)/backend/data/models/stone_detector.onnx

validate: ## Run pipeline against val dataset and print report
	uv --directory backend run python -m goapp.cli.compare_on_val \
		--val-dir $(VAL_DIR) --model $(MODEL) --status accepted --verbose

export-models: ## Export trained .pt weights to ONNX for the lean serving image
	uv --directory backend run --extra ml python scripts/export_onnx.py

# ---------------------------------------------------------------------------
# Local PDF ingest (offload board detection from Cloud Run)
# ---------------------------------------------------------------------------

# Run as: make ingest-pdf USER=pwsiegel@gmail.com PDF=/path/to/book.pdf
# Detects boards on the laptop, writes problems under $GOAPP_DATA_DIR, then
# rsyncs the user's tsumego/ directory to the GCS bucket. Pass NO_UPLOAD=1
# to skip the upload step.
ingest-pdf: ## Detect boards on a PDF locally and sync to GCS (USER=email PDF=path)
	@if [ -z "$(USER)" ] || [ -z "$(PDF)" ]; then \
		echo "usage: make ingest-pdf USER=email-or-id PDF=/path/to/file.pdf"; \
		exit 2; \
	fi
	uv --directory backend run python -m goapp.cli.ingest_pdf \
		--user "$(USER)" --pdf "$(PDF)" $(if $(NO_UPLOAD),--no-upload,)

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-up: ## Build + run the single-image container (frontend + API on :8080)
	docker compose up --build

docker-down: ## Stop the docker compose service
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

modal-gen-synth: ## Wipe + regenerate synth_pages on the Modal volume (CPU, ~3 min)
	modal run training/modal_train.py::gen_synth

modal-train-smoke: ## Run a tiny ~2 min smoke test on Modal (L4)
	modal run training/modal_train.py::smoke

modal-train-boards: ## Train the board detector on Modal (L4)
	modal run training/modal_train.py::train_boards

modal-train-stones: ## Train the stone detector on Modal (L4)
	modal run training/modal_train.py::train_stones

modal-pull-weights: ## Copy trained weights from the Modal volume into backend/data/models/ (skips any not yet on the volume)
	@for f in board_detector.pt stone_detector.pt; do \
		echo "==> $$f"; \
		modal volume get --force $(MODAL_VOLUME) /models/$$f backend/data/models/ \
			|| echo "   (skipped — not on volume)"; \
	done
