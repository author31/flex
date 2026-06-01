# Implementation Plan: Prompt-Guided Facial Expression Editing for Animated Characters

**Branch**: `001-facial-expression-editing` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-facial-expression-editing/spec.md`

**Related**: [research.md](./research.md) (model selection), [design.md](./design.md) (API/UI/deploy)

## Summary

A monorepo prototype that edits facial expressions on animated-character images from
a text prompt, confined to a user-selected face region, while preserving identity.
Backend (FastAPI, DDD 4-layer) runs three swappable model jobs — anime face
segmentation (mask), SDXL inpainting (edit), open_clip (evaluation) — and composites
the edit back onto the original through the mask so off-mask pixels are byte-identical.
Frontend (React + Vite) drives upload → region select → prompt → compare-with-metrics.
One-click `docker compose up` with hot-mounted packages.

## Technical Context

**Language/Version**: Backend Python 3.12; Frontend TypeScript (React 18 + Vite 5)

**Primary Dependencies**:
- Backend: FastAPI, uvicorn, `diffusers`, `transformers`, `torch` (CUDA 12.4),
  `open_clip_torch`, `pillow`, `numpy`, `pydantic`; managed by `uv`
- Frontend: react, react-dom, vite, typescript, @tanstack/react-query
- Models: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (default inpaint),
  Anime-Face-Segmentation (siyeong0) U-Net, open_clip ViT-L/14

**Storage**: Filesystem (`/data` volume) — image bytes, masks, results + JSON
metadata sidecars. No DB for the prototype (repository port allows DB later).

**Testing**: Backend `pytest` (domain unit, application use-case, API contract);
Frontend `vitest` + `@testing-library/react`; integration test across the
backend↔frontend HTTP contract.

**Target Platform**: Linux server with NVIDIA GPU (containerized via Docker Compose).

**Project Type**: Web application — monorepo `backend/` + `frontend/`.

**Performance Goals**: Single-edit end-to-end under 2 min (SC-003); SDXL inpaint
~15–30 steps, target < ~30 s on GPU. Async job model so the API never blocks.

**Constraints**: Off-mask identity 100% preserved (SC-002) via mask compositing;
expression edit success ≥ 80% (SC-001); CLIP-in-mask agreement ≥ baseline for ≥ 80%
(SC-004). No committed secrets; GPU required for usable latency.

**Scale/Scope**: Single-user, single-image prototype. 4 expression presets + 4
regions (face/eyes/mouth/eyebrows). ~10 endpoints, ~6 frontend components.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Monorepo Boundaries | `backend/` + `frontend/` only; coupling over HTTP only | PASS — two packages, REST contract only |
| II. Backend Layered Arch (NON-NEGOTIABLE) | EXACTLY `domain.py`, `application.py`, `repository.py`, `infrastructure.py`; inward deps; pure domain | PASS — see Project Structure; model engines are ports in application, impls in infrastructure |
| III. Test-First (NON-NEGOTIABLE) | tests written first, per-layer + integration, no skipped | PASS — pytest+vitest plan, contract tests from openapi |
| IV. Code Quality | Py3.12 typed + uv + lint/type-check; React+Vite+ESLint warning-clean | PASS |
| V. Containerized Parity | Docker Compose single source of truth; env config; no secrets | PASS — compose + Dockerfiles + .env.example, `.env` gitignored |

**Initial gate**: PASS. No violations → Complexity Tracking empty.

**Post-design re-check**: PASS — data-model entities live in `domain.py`; contracts map
to `infrastructure.py` routers; no new layer files introduced.

## Project Structure

### Documentation (this feature)

```text
specs/001-facial-expression-editing/
├── plan.md              # This file
├── spec.md              # Feature spec (WHAT/WHY)
├── research.md          # Phase 0 — model selection (done)
├── design.md            # API/UI/deploy design (done)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (openapi.yaml)
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
backend/
├── app/
│   ├── domain.py            # entities, value objects, enums, rules (pure, no I/O)
│   ├── repository.py        # persistence Protocols + filesystem impls (→ domain)
│   ├── application.py       # use cases + model ports (→ domain, repository)
│   └── infrastructure.py    # FastAPI app/routers/schemas + model adapters + DI + config
├── tests/
│   ├── unit/                # domain rules
│   ├── application/         # use cases (fake ports/repos)
│   └── contract/            # API contract tests vs contracts/openapi.yaml
├── pyproject.toml           # uv-managed
└── Dockerfile               # (exists)

frontend/
├── src/
│   ├── api/                 # typed client (VITE_API_BASE)
│   ├── hooks/               # useUpload, useSegments, useCreateEdit, useEditPolling
│   ├── components/          # ImageUploader, RegionSelector(+RegionChips,BrushTool),
│   │                        #   PromptPanel, SubmitBar, ComparisonView(+MetricsPanel)
│   ├── pages/EditorPage.tsx
│   └── main.tsx
├── tests/                   # vitest + testing-library
├── package.json
├── vite.config.ts           # dev proxy /api → backend:8000
└── Dockerfile               # (exists)

docker-compose.yml           # (exists) one-click, hot mount, GPU
.env.example                 # (exists)
```

**Structure Decision**: Web-application monorepo (constitution Principle I). Backend
uses the exact 4-file DDD layout (Principle II) under `backend/app/`; tests mirror the
layers. Frontend is feature-component organized under `frontend/src/`. Both ship a
Dockerfile; `docker-compose.yml` at root is the single deploy source of truth.

## Complexity Tracking

> No constitution violations. Section intentionally empty.
