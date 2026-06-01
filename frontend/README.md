# Flex Frontend

React + Vite + TypeScript UI for the facial expression editor. Monochrome theme
(`src/styles/theme.css`). State/data via TanStack Query.

## Flow

Upload (`ImageUploader`) → select region (`RegionSelector`: auto `RegionChips` or
`BrushTool`) → prompt/preset (`PromptPanel`) → submit (`SubmitBar`) → poll →
before/after + metrics (`ComparisonView` + `MetricsPanel`).

## Run

```bash
npm install
npm run dev        # http://localhost:5173 (proxies /api → backend:8000)
```

## Test

```bash
npm run test       # vitest + testing-library (component + flow tests, mocked API)
```

## Config

`VITE_API_BASE` (default `/api`) — base for the typed client in `src/api/client.ts`.
