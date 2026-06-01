# Quickstart

**Feature**: 001-facial-expression-editing

One-click local stack via Docker Compose (constitution Principle V). Requires Docker
with the NVIDIA Container Toolkit (GPU) for usable SDXL latency.

## Run

```bash
cp .env.example .env        # fill HUGGING_FACE_HUB_TOKEN only if using gated models
docker compose up           # builds backend (CUDA) + frontend, hot-mounts both
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/api/v1 (health: `/api/v1/health`)
- First boot pulls models into the `hf-cache` volume (slow once; `start_period` 120s).

Hot reload: editing `backend/` (uvicorn `--reload`) or `frontend/` (vite HMR) updates
live without rebuild.

## Smoke test (API)

```bash
# 1. upload
curl -F file=@character.png http://localhost:8000/api/v1/images
# -> {"image_id":"img_abc", ...}

# 2. auto-segment regions
curl -X POST http://localhost:8000/api/v1/images/img_abc/segments

# 3. create edit (preset smile on mouth)
curl -X POST http://localhost:8000/api/v1/edits \
  -H 'content-type: application/json' \
  -d '{"image_id":"img_abc","region":"mouth","preset":"smile"}'
# -> 202 {"edit_id":"edit_xyz","status":"processing"}

# 4. poll until completed
curl http://localhost:8000/api/v1/edits/edit_xyz

# 5. comparison + metrics
curl http://localhost:8000/api/v1/edits/edit_xyz/comparison
```

## UI flow

Upload image → pick region chip (Face/Eyes/Mouth/Eyebrows) or brush a custom mask →
choose a preset (Smile/Angry/Sad/Surprised) or type a prompt → Submit → watch the
before/after slider and the metrics panel (CLIP-in-mask, edit success, identity
preserved, latency).

## Acceptance mapping

| Check | How |
|-------|-----|
| FR-001 upload | step 1 |
| FR-002 region | step 2 / brush |
| FR-003/006/007 prompt+preset+detail | step 3 |
| FR-004/005 confined edit + identity | composited result; `identity_preserved=true` |
| FR-008 validation | empty prompt / no region → 422 |
| FR-009 compare+retry | step 5 + new submit |
| FR-010 metrics | step 5 metrics block |

## Dev (without Docker)

```bash
# backend
cd backend && uv sync && uv run uvicorn app.infrastructure:app --reload
# frontend
cd frontend && npm install && npm run dev
```

## Tests

```bash
cd backend && uv run pytest          # unit (domain) + application + contract
cd frontend && npm run test          # vitest component tests
```
