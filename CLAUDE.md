# Collaboration notes

- Discuss the plan before making big changes (new projects, scaffolding, installs, architectural shifts, anything that touches many files). Propose the approach, iterate in conversation, and wait for a clear go-ahead before executing.
- Don't ask for approval before making small changes (single-file edits, small refactors, fixing an obvious bug, adding a missing import). Just do them.
- When unsure which bucket a change falls into, err toward action — the user will interrupt if needed.

# Scope of autonomy

- Inside this project folder (`/Users/paul/pwsiegel/tsumego`): do whatever you want. Install, delete, refactor, run scripts. Nothing here is precious.
- Outside this folder (the rest of the computer, system settings, other projects, shell/user config): ask first.
- Web search is fine for project-related research. No sketchy sites — stick to reputable docs, official repos, and established sources.

# Project layout

- `web/` — React + TS + Vite frontend. Dev server on :5173. Vite proxies `/api/*` to the backend.
- `backend/` — FastAPI backend managed by `uv`. Dev server on :8001 (not 8000 — something else on this machine owns 8000). Run: `uv --directory backend run uvicorn goapp.api:app --reload --port 8001`.
- `docker-compose.yml` — brings up both (web on :8080, api on :8001) for integration-style local dev.
- Deployment target: Cloudflare Pages (frontend) + Cloud Run (backend).
