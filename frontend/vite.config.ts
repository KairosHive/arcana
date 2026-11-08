import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// The built bundle is served by Arcana's own Flask server from
// arcana/assets/ui, so paths are relative and the output lands there.
// In development `vite` runs on 5173 and forwards everything that is not a
// page to the Python server on 8051 (ARCANA_DEV_PORT).
const backend = `http://127.0.0.1:${process.env.ARCANA_DEV_PORT || '8051'}`

export default defineConfig({
  plugins: [svelte()],
  base: './',
  build: {
    outDir: '../arcana/web',
    emptyOutDir: true,
    // Plotly and xterm are dynamic imports, so they land in chunks of their own.
    chunkSizeWarningLimit: 4500,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ['/api', '/ops', '/thumb', '/preview', '/palette', '/audio', '/awave', '/aspec',
       '/board-page', '/agent'].map((p) => [p, { target: backend, changeOrigin: false }]),
    ),
  },
})
