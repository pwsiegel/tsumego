import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base` is the path the app is served from. GitHub Pages project sites live
// under /<repo>/, so set VITE_BASE=/go-training/ for a production Pages build;
// defaults to '/' for local dev and custom domains.
//
// The Firebase SDK is the bulk of the bundle; split it (and the React runtime)
// into their own chunks so the app code stays small and the heavy vendor chunks
// cache across deploys.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
  // The KataGo engine runs in a module worker that imports TensorFlow.js.
  // Bundle the worker as ESM and pre-bundle tfjs so the worker doesn't stall
  // loading a long unbundled ESM chain in dev.
  worker: { format: 'es' },
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgpu',
      'pako',
    ],
  },
  server: {
    // Fail loudly if 5173 is taken rather than sliding to 5174 (which breaks
    // Firebase Storage CORS, since only :5173 is in the bucket allowlist).
    port: 5173,
    strictPort: true,
    proxy: {
      // Proxy the local KataGo bridge to the backend dev server so the app can
      // call it same-origin (no CORS). Only active in `vite dev`; production
      // builds have no engine to reach. Override with VITE_KATAGO_TARGET.
      '/api/katago': {
        target: process.env.VITE_KATAGO_TARGET ?? 'http://localhost:8001',
        changeOrigin: true,
      },
      // Fox game sync — also backend-only (the Fox API sends no CORS headers),
      // so it can't run in the deployed app. Dev-proxied like KataGo.
      '/api/fox': {
        target: process.env.VITE_KATAGO_TARGET ?? 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        advancedChunks: {
          groups: [
            { name: 'firebase', test: /[\\/]node_modules[\\/]@?firebase[\\/]/ },
            { name: 'react', test: /[\\/]node_modules[\\/](react|react-dom|react-router|react-router-dom|scheduler)[\\/]/ },
          ],
        },
      },
    },
  },
})
