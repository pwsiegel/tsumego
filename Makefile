.PHONY: help setup api web dev lint \
       synth train-boards train-stones validate \
       docker-up docker-down

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
MODEL   ?= $(HOME)/data/go-app/models/stone_detector.pt

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
