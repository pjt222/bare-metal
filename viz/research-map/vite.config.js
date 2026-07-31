import { defineConfig } from 'vite';

// The knowledge graph is published as a subdirectory of the documentation site
// (#192), which is a GitHub *project* page served from /bare-metal/ — not from a
// domain root. Without a matching `base`, Vite emits absolute asset paths like
// /assets/index-*.js, which 404 under that prefix. That is the same mechanism
// that made this app's source index.html unservable when Quarto copied it into
// the site during #187.
//
// Switched on GITHUB_ACTIONS so a local `npm run dev` / `npm run build` keeps
// working from the filesystem root, matching the pattern the sibling project
// ez-ar2diff uses in its viz/vite.config.js.
const PAGES_BASE = '/bare-metal/viz/research-map/';
const isGitHubPages = process.env.GITHUB_ACTIONS === 'true';

export default defineConfig({
  base: isGitHubPages ? PAGES_BASE : '/',
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: 'dist',
    // Sourcemaps locally, not in the published bundle. The map is 1.99 MB
    // against a 490 KB bundle — 80% of what a reader downloads — and the
    // unminified source it reconstructs is a click away in the repository this
    // page links to. Keep them for local debugging, where that trade does not
    // apply.
    sourcemap: !isGitHubPages,
  },
});
