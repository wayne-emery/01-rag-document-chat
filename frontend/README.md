# Frontend &mdash; RAG Document Chat

Vite + React 19 + TypeScript + Tailwind CSS v4.

> See the [project README](../README.md) for the full pitch and the [architecture doc](../docs/architecture.md) for design rationale (including why this is a 3-pane layout).

## Setup

```bash
npm install
npm run dev
```

Opens at http://localhost:5173. Requires the backend running at http://127.0.0.1:8000 (the Vite dev server proxies `/api/*` to it).

## Configuration

Copy `.env.example` to `.env` only if you need to override the API base URL (e.g. for a deployed backend).

## Scripts

- `npm run dev` — start dev server with HMR
- `npm run build` — type-check + production build
- `npm run preview` — preview the production build locally
- `npm run lint` — run ESLint

## Layout

```
src/
├── App.tsx                  # three-pane shell + health/document state
├── main.tsx                 # entry point
├── index.css                # Tailwind v4 import + minor base styles
├── types.ts                 # shared types mirroring the backend schemas
├── api/client.ts            # typed fetch wrappers for the backend endpoints
└── components/
    ├── UploadPanel.tsx      # left pane — drag-and-drop upload + document list + clear
    ├── ChatPanel.tsx        # center pane — message bubbles, [N] source pills, input
    └── SourcesPanel.tsx     # right pane — ranked chunks, score badges, jump-to highlight
```
