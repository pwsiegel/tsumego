.PHONY: help setup api web dev lint \
       synth train-boards train-stones validate \
       docker-up docker-down \
       deploy logs \
       sync-synth build-training-image \
       train-cloud-smoke train-cloud-boards train-cloud-stones pull-weights

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
# Vertex AI training (L4 spot)
# ---------------------------------------------------------------------------

GCP_BUCKET         := $(GCP_PROJECT)-data
GCP_TRAINING_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/apps/training:latest
SYNTH_LOCAL_DIR    := $(HOME)/data/go-app/data/synth_pages
SYNTH_GCS_DIR      := gs://$(GCP_BUCKET)/data/synth_pages
MODELS_GCS_DIR     := gs://$(GCP_BUCKET)/data/models

sync-synth: ## Upload local synth_pages/ to GCS (run after `make synth`)
	gsutil -m rsync -d -r $(SYNTH_LOCAL_DIR) $(SYNTH_GCS_DIR)

build-training-image: ## Build the CUDA training image via Cloud Build
	gcloud builds submit \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--config=training/cloudbuild.yaml \
		--substitutions=_IMAGE=$(GCP_TRAINING_IMAGE) .

train-cloud-smoke: ## Submit a tiny ~2 min Vertex job to verify the pipeline
	gcloud ai custom-jobs create \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--display-name=smoke-$(shell date +%Y%m%d-%H%M%S) \
		--config=training/job-smoke.yaml

train-cloud-boards: ## Submit board-detector training to Vertex (L4 spot)
	gcloud ai custom-jobs create \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--display-name=board-detector-$(shell date +%Y%m%d-%H%M%S) \
		--config=training/job-boards.yaml

train-cloud-stones: ## Submit stone-detector training to Vertex (L4 spot)
	gcloud ai custom-jobs create \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--display-name=stone-detector-$(shell date +%Y%m%d-%H%M%S) \
		--config=training/job-stones.yaml

pull-weights: ## Copy trained weights from GCS into backend/data/models/
	gsutil cp $(MODELS_GCS_DIR)/board_detector.pt backend/data/models/
	gsutil cp $(MODELS_GCS_DIR)/stone_detector.pt backend/data/models/
