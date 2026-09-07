# go-training

The public, user-facing static web app for solving tsumego (Go life-and-death problems) and reviewing them with a teacher. React + Vite + TypeScript, deployed to GitHub Pages.

Problems and all user data live in Firebase (auth-gated Storage + Firestore); this repo holds only the app code. The tooling that produces that data — PDF scanning, ML stone/board detection, the KataGo analysis API, and the ingest/export pipeline — lives in a separate private repo, **`pwsiegel/go-training-tools`** (which nests this repo at `app/` locally).

## Develop

```sh
npm install
cp .env.example .env    # fill in VITE_FIREBASE_* (Firebase console / the private repo)
npm run dev             # http://localhost:5173
npm run build
npm run lint
```

The **game board** — one surface with a **Play / Review** toggle — is always available and ships to Pages, with **KataGo analysis that runs entirely in the browser** — TensorFlow.js / WebGPU, no backend. AI engine settings are site-wide: a status button at the bottom of the sidebar shows the loaded net and its health (ready / warming / down / in use by another tab) and opens the shared settings modal — playouts and GPU batch (auto-tuned to the device, with a manual override). Analysis is session-style: the position you're on streams snapshots as the search deepens, a play/pause button keeps it deepening ("ponder"), and a counter shows the playouts behind what's on screen; navigating one move re-roots the existing search instead of restarting. Net weights are served from Firebase Storage. Click any point on the board to branch **variations** — the same in-browser AI evaluates them like any other position — and they persist per user in a private `reviews/` object (separate from the game, so a student and a teacher keep independent variations) that reloads automatically. Because the in-browser engine drives the GPU, only one tab or window may run AI at a time — a Web Lock coordinates across them (both modes share it), and a second one waits with a note explaining why.

In **play** mode the human-like net answers your moves, and it answers only at the leaf of the line you're on — so stepping back and playing something else branches a variation rather than truncating the game, and a game played out from a reviewed position is stored as a variation of it like any other. You can switch sides at any point; the opponent takes over the colour you leave. Play never analyzes: the score you see obeys the score setting (show / hide / alert on falling behind), drawn from the human net's own estimates. Switching to review turns the AI on, swapping the resident net (a few seconds). Opponent settings — net, rank, sharpness, move delay, score display — sit in the shared AI settings modal beside the analysis settings, and all of them apply mid-game.

Games come from three sources:

- **Fox** — imported from [Fox Weiqi](https://www.foxwq.com/) by a local-only sync (the Fox API sends no CORS headers, so it can't run in the deployed app). Synced games persist to Firestore and are reviewable here on GitHub Pages.
- **Play** — games played here against a human-like KataGo at a chosen rank. The opponent is the human-SL net (`b18c384nbt-humanv0`) run **in the browser** (WebGPU) — no backend — so play ships to Pages too. A game started from an empty board (**Play** in the sidebar) is session-only until you **Save** it: the first line you played becomes the game, any branches off it become its variations.
- **Upload** — SGFs added by hand from the games list.

The **"explore"** tsumego hints also run in the browser, so nothing analysis-related needs a backend. The private backend (`/api/katago`, started with `make api`, which also serves the Fox sync) is optional: when it's running, the AI engine modal offers a full-strength **native KataGo (Metal)** model, with the same streaming session analysis via a GTP bridge (`/api/katago/session`). The dev server proxies `/api/katago` and `/api/fox` to it.

## Deploy

Pushing to `main` runs `.github/workflows/deploy-pages.yml`, which builds and publishes to GitHub Pages. Firebase web config (public values) is injected from repository **Variables** (`VITE_FIREBASE_*`, and `VITE_BASE=/go-training/`).
