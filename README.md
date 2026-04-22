# Go Problem Workbook

Interactive workbook for annotating Go problems: import from PDF, detect the board position, work out the solution as a numbered move sequence.

## Layout

```
go-app/
├── web/                React + TypeScript + Vite frontend
├── api/                FastAPI backend (PDF ingestion, detection, training)
└── docker-compose.yml  Local dev: brings up both
```

## Local dev

Two terminals:

```bash
# backend (port 8001)
cd api
uv sync
uv run uvicorn goapp_api.main:app --reload --port 8001
```

```bash
# frontend (port 5173; proxies /api → backend)
cd web
npm install
npm run dev
```

Open http://localhost:5173.

Or with Docker:

```bash
docker compose up --build
# frontend on :8080, backend on :8001
```

## Deployment target

Intended: Cloudflare Pages (static frontend) + Cloud Run (containerized backend).
See individual service Dockerfiles.

## Roadmap

Tracks ideas as they come up. Ordering within a section is rough priority.

### Play mode
- [ ] **Under-the-stones moves** — allow placing a move on an intersection that already holds a stone, provided the prior stone has been captured in the evolving position. Requires capture/liberty logic.
- [ ] **Multi-number annotation** — a single intersection may hold several move numbers over the life of a problem (move N played there, captured, move M played there later). Render stacked or comma-separated labels.
- [ ] **Illegal-move detection** — warn on suicide / ko violation / playing on an occupied point that hasn't been captured.
- [ ] Branching variations (tree of moves, not just a linear sequence).

### PDF import / detection
- [ ] Auto-detect corner placement (row/column intensity peaks to pre-place the 4 handles).
- [ ] **Correction UI**: click a detected stone to cycle Empty → B → W → Empty. Also serves as the labeling tool.
- [ ] Export labeled patches as training data (JSON + image tiles).
- [ ] Tiny CNN patch classifier trained in PyTorch, exported to ONNX, loaded via onnxruntime-web.
- [ ] Batch-ingest an entire PDF in one call (backend endpoint returning all detected problems).
- [ ] **Problem-label detection (OCR)** — crop a region around each detected board and OCR out the original book's label ("problem 7", "問題 3", "第 12 題", etc.) so imported problems keep their original numbering instead of sequential indices. Multilingual Tesseract is the obvious starting point, but accuracy craters on low-DPI photocopy scans — needs a fallback and probably per-book heuristics for where the label sits relative to the board.
- [ ] **Stone-annotation detection** — books frequently print a number, triangle/square/circle mark, or a letter/kanji inside a stone to identify "move 7" / "the marked stone" / "stone A". Detecting these would let us (a) transcribe the book's annotations accurately into the SGF (`LB[]`, `TR[]`, numbered move sequences) rather than collapsing everything to a setup position, and (b) disambiguate stones from nearby printed characters — observed on hm2 where a Korean character near the board was misread as a white stone by the detector. Plausible approach: after stone detection, crop each stone's interior and run a small CNN or OCR pass for marks/digits/characters; also use the presence of detected characters outside stones as a negative signal to suppress near-board text false positives.

### Solving & analysis

The obvious primitive is KataGo — strongest open-source Go engine, MIT-licensed, runs locally. The non-obvious problem: KataGo evaluates the *whole board*. Drop a corner tsumego onto an otherwise empty 19×19 and its policy will open a different corner, because that's the best move on this global position. Making it useful for local problems is real work.

Candidate approaches (not committed to one yet):

- [ ] **Local fencing.** Wall off the tactical region with thick stones on the owner's side; engine attention concentrates locally. Cheapest to prototype; breaks on problems where outside influence matters.
- [ ] **Local search + KataGo as evaluator.** Don't trust KataGo's policy; use its *value* head only. Generate candidate moves within a bounded active region, alpha-beta or MCTS over local moves, KataGo scores terminal positions. How real tsumego solvers work. More effort, more robust.
- [ ] **Fine-tune on tsumego.** Continue-train KataGo (or a smaller net) on a labeled tsumego dataset (~few hundred thousand problems exist publicly). Net learns "in problem-like positions, answer locally." Most ML work; generalizes best across problem types.
- [ ] **Dedicated tsumego solvers (prior art).** Proof-number search, GnuGo's life/death reader, various academic tools. Usually narrower in scope. Worth surveying before reinventing.

Downstream tasks that depend on picking one:

- [ ] **Check submitted solutions** — given an ingested problem, detect the intended outcome (black lives / black kills / connects / …) and evaluate whether a submitted sequence achieves it.
- [ ] **Auto-generate solutions** — principal variation from the solver, rendered as numbered moves.
- [ ] **Problem categorization** — classify by type (life-and-death, tesuji, connection, endgame) and by motif (ladder, net, throw-in, snapback…). No off-the-shelf tool exists. The global-vs-local concern is lighter here since categorization cares about *shape*, not best-move. Approach: embed the position with a pretrained Go net, cluster, hand-label the clusters. Becomes a tag taxonomy on each ingested problem.

Open infra question: bundle KataGo (adds ~100MB and a C++ dependency) vs. run it as a separate service.

### Persistence / export
- [ ] Save/load individual problems (localStorage at minimum).
- [ ] SGF export (standard Go file format, interoperable with every Go program).
- [ ] PNG export of the annotated board.

### Theming
- [ ] Board/stone themes — CSS custom properties + `<defs>` for gradients, shadows, wood textures, slate/shell stones.

### Infra / deploy
- [ ] CI: typecheck + test on PR.
- [ ] Deploy frontend to Cloudflare Pages.
- [ ] Deploy backend to Cloud Run.
