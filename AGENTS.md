# Repository Guidelines

## Project Structure & Module Organization
- Root workspace with npm workspaces; code lives under `packages/`.
- Key modules:
  - `packages/viewer` (Svelte UI), `packages/component` (shared UI), `packages/table` (table widget),
    `packages/embedding-atlas` (wrapper exports), `packages/backend` (widgets/integration),
    `packages/umap-wasm` and `packages/density-clustering` (WASM + algorithms),
    `packages/examples` (demo apps), `packages/docs` (VitePress docs).
- Data for local runs may be created by examples; scripts live in `scripts/`.

## Build, Test, and Development Commands
- Root build: `npm run build` — builds all packages and docs via `scripts/build.sh`.
- Root tests: `npm run test` — runs package tests (vitest) where defined.
- Format check: `npm run check-format` — checks Prettier formatting.
- Per-package:
  - Dev server: `npm run dev` (e.g., in `packages/viewer`).
  - Package build: `npm run package` (build + publint for publishable libs).
  - Preview: `npm run preview` where available.

## Coding Style & Naming Conventions
- Languages: TypeScript/JavaScript, Svelte, Rust (WASM crates).
- Indentation: 2 spaces; use Prettier (`.prettierrc`) with `prettier-plugin-svelte`.
- Filenames: kebab-case for assets and Svelte files, camelCase/PascalCase for TS/JS identifiers.
- Keep components small and typed; prefer explicit types for public APIs.

## Testing Guidelines
- Framework: `vitest` in algorithmic packages (e.g., `packages/umap-wasm`, `packages/density-clustering`).
- Add focused unit tests near implementation; name files `*.test.ts`.
- Run tests from root (`npm run test`) or within a package (`npm run test`).

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; scope in parentheses when helpful.
  - Example: `viewer: guard undefined identifier in animation`.
- PRs: include purpose, approach, and screenshots/GIFs for UI changes; link issues.
- Keep changes scoped; update docs or examples if behavior changes.

## Security & Configuration Tips
- Avoid committing large datasets or secrets; prefer environment-agnostic examples.
- WASM builds can be heavy; run only the targets you need during development.
