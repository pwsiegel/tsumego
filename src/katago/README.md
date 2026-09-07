# Vendored in-browser KataGo engine

`engine/`, `utils/`, and `types.ts` are vendored from
[Sir-Teo/web-katrain](https://github.com/Sir-Teo/web-katrain) (MIT — see `LICENSE`),
a browser-native KataGo pipeline built on TensorFlow.js. It parses KataGo
`.bin.gz` weights, extracts v7 input features, runs forward passes, and does
PUCT/MCTS search — entirely client-side, no backend.

**This copy runs on WebGPU only.** Upstream falls back to WASM and then CPU when
WebGPU is missing; that path is deleted here, because the net needs ~12x longer
per evaluation on the CPU and cannot finish a useful search, so a fallback only
produces a slow wrong answer. `initBackend` asserts the active backend is
`webgpu` and throws otherwise, which surfaces as a red engine light in the app.

`webEngine.ts` is ours: a thin wrapper that bridges the app's game types
(`Stone` / `GameMove`) to the engine and returns a trimmed analysis.

## Notes
- The worker is pre-bundled by `../../scripts/build-worker.mjs` into
  `public/katago-worker.js` (rebuilt on every `dev`/`build`), so it runs
  identically under `vite dev` and the production build.
- The model net (`public/models/*.bin.gz`) and the generated worker are
  gitignored; the net still needs hosting for the deployed build.

## Updating from upstream
Re-copy `src/engine/katago`, `src/types.ts`, and the used `src/utils/*` files,
preserving this directory layout so the relative imports (`../../types`,
`../../utils/*`) keep resolving. Then re-apply the WebGPU-only change above:
upstream will bring back the WASM/CPU fallback and a selectable backend.
