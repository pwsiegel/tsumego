// Pre-bundle the KataGo tfjs worker into a single self-contained file.
//
// Vite dev serves modules as raw ESM, which the tfjs worker doesn't survive;
// the production build bundles it via Rollup, which works. This produces that
// same bundled worker up front so BOTH dev and prod load one static file
// (public/katago-worker.js) — the engine runs identically in either.

import { build } from 'esbuild';

const base = process.env.VITE_BASE || '/';

await build({
  entryPoints: ['src/katago/engine/katago/worker.ts'],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
  outfile: 'public/katago-worker.js',
  define: { 'import.meta.env': JSON.stringify({ BASE_URL: base }) },
  logLevel: 'info',
});

console.log(`built public/katago-worker.js (base=${base})`);
