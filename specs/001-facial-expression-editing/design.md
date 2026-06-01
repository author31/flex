# Technical Design: Backend + Frontend + Deploy

**Feature**: 001-facial-expression-editing
**Date**: 2026-06-01
**Depends on**: [research.md](./research.md), [spec.md](./spec.md), constitution

Grounds the spec into a concrete monorepo: FastAPI backend (DDD 4-layer), React+Vite
frontend, one-click Docker Compose with hot mounts.

---

## 1. Architecture

```
                    docker compose up
  ┌──────────────────────────┐        ┌──────────────────────────────┐
  │ frontend (React+Vite)    │  HTTP  │ backend (FastAPI, GPU)        │
  │ upload · mask · prompt   │ ─────▶ │ /api/v1/...                   │
  │ comparison + metrics     │ ◀───── │ segment · inpaint · evaluate  │
  └──────────────────────────┘  JSON  └──────────────────────────────┘
        :5173 (vite dev)                       :8000 (uvicorn)
   bind mount ./frontend                  bind mount ./backend
                                          volumes: hf-cache, app-data
```

Edit flow: upload image → (auto-segment | manual brush) → choose prompt/preset →
POST edit job → backend inpaints inside mask → composites onto original → evaluates
→ frontend polls job → shows original vs edited + metrics.

---

## 2. Backend — DDD 4-layer mapping (constitution: EXACTLY these 4 files)

`backend/app/` — `domain.py`, `application.py`, `repository.py`, `infrastructure.py`.
Dependency flow: `infrastructure → application → domain`; `repository → domain`.
`domain.py` imports none of the others. Model engines are **ports** declared in
`application.py` (Protocols) and implemented in `infrastructure.py`.

### domain.py (pure, no I/O)
- Value objects / enums:
  - `FacialRegion` = {FACE, EYES, MOUTH, EYEBROWS, CUSTOM}
  - `ExpressionPreset` = {SMILE, ANGRY, SAD, SURPRISED}
  - `Mask` (binary mask geometry + bbox; ops: `is_empty`, `bbox`, `area`)
  - `EditParams` (strength, steps, guidance, seed) with validated ranges
- Entities: `CharacterImage`(id, dims), `EditJob`(id, image_id, mask, prompt,
  params, status), `EditResult`(id, job_id, result_image_ref), `EvaluationRecord`
  (clip_similarity_in_mask, edit_success, edit_success_score, identity_preserved,
  latency_ms)
- Rules (raise domain errors): prompt non-empty; region/mask required & non-empty;
  preset∈supported; params in range. `preset_to_prompt(preset, region)` mapping.

### repository.py (persistence contracts + impls, depends on domain only)
- Protocols: `ImageRepository`, `JobRepository`, `ResultRepository`
- Default impl: `FileSystemImageRepository` etc. — store image bytes + masks under
  `app-data/`, metadata as JSON sidecars. (Swappable for DB later.)

### application.py (use cases; depends on domain + repository + ports)
- Ports (Protocols implemented by infrastructure):
  - `Segmenter.segment(image) -> dict[FacialRegion, Mask]`
  - `Inpainter.inpaint(image, mask, prompt, params) -> Image`
  - `Evaluator.evaluate(before, after, mask, prompt) -> EvaluationRecord`
- Use cases:
  - `ImportImageUseCase`
  - `SegmentImageUseCase` (auto masks; eyebrows derived from eye band)
  - `GenerateEditUseCase` (resolve mask → inpaint → **composite onto original
    using mask** → persist → evaluate → persist) — guarantees off-mask identity
  - `GetEditUseCase` / `GetComparisonUseCase`
  - `ListPresetsUseCase`

### infrastructure.py (FastAPI + adapters + DI + config)
- FastAPI app, `APIRouter`s, Pydantic request/response schemas, exception handlers
  mapping domain errors → HTTP 4xx
- Adapters: `DiffusersInpainter` (AutoPipelineForInpainting, `MODEL_ID` env),
  `AnimeFaceSegmenter` (siyeong0 U-Net), `OpenClipEvaluator` (ViT-L/14)
- Config via env (`Settings`), device/GPU selection, model warm-load on startup,
  composition helper (alpha-blend at mask edge)

## 3. RESTful API (`/api/v1`)

Resource-oriented, JSON; images returned as URLs (served from `app-data`). Edits
are async jobs (202 + poll) so SDXL latency never blocks the request.

| Method | Path | Purpose | Body / Returns |
|--------|------|---------|----------------|
| GET | `/api/v1/health` | liveness/readiness (models loaded) | `{status, device, model_id}` |
| GET | `/api/v1/presets` | supported presets + regions | `{expressions:[...], regions:[...]}` |
| POST | `/api/v1/images` | upload image (multipart `file`) | 201 `{image_id, width, height, url}` |
| GET | `/api/v1/images/{id}` | image metadata | `{image_id, width, height, url}` |
| GET | `/api/v1/images/{id}/file` | raw image bytes | image/png |
| POST | `/api/v1/images/{id}/segments` | run auto face segmentation | 200 `{regions:{face:{url,bbox,area}, eyes:{...}, mouth:{...}, eyebrows:{...}}}` |
| GET | `/api/v1/images/{id}/segments/{region}/file` | region mask png | image/png |
| POST | `/api/v1/edits` | create edit job | 202 `{edit_id, status:"processing"}` |
| GET | `/api/v1/edits/{id}` | job status + result refs | `{edit_id, status, result?, metrics?}` |
| GET | `/api/v1/edits/{id}/comparison` | original+edited+metrics | see below |
| GET | `/api/v1/edits/{id}/result/file` | edited image bytes | image/png |

### POST /api/v1/edits — request
```json
{
  "image_id": "img_abc",
  "region": "mouth",                 // FacialRegion; OR provide "mask"
  "mask": "data:image/png;base64,…", // optional manual brush mask (white=edit)
  "prompt": "a wide happy smile",    // free-form; OR "preset"
  "preset": "smile",                 // optional; mapped to prompt server-side
  "params": { "strength": 0.99, "steps": 25, "guidance": 8.0, "seed": 1234 }
}
```
Rules: exactly one of `region`|`mask` required; exactly one of `prompt`|`preset`
required. Empty/invalid → 422 with domain error message (FR-008).

### GET /api/v1/edits/{id}/comparison — response
```json
{
  "edit_id": "edit_xyz",
  "status": "completed",
  "original": { "image_id": "img_abc", "url": "/api/v1/images/img_abc/file" },
  "edited":   { "image_id": "img_def", "url": "/api/v1/edits/edit_xyz/result/file" },
  "region": "mouth",
  "prompt": "a wide happy smile",
  "metrics": {
    "clip_similarity_in_mask": 0.31,
    "edit_success": true,
    "edit_success_score": 0.12,
    "identity_preserved": true,
    "latency_ms": 4200
  }
}
```

Maps to requirements: FR-001 (POST images), FR-002 (segments + mask), FR-003/006/007
(prompt/preset/region on POST edits), FR-004/005 (inpaint+composite), FR-008
(422 validation), FR-009 (comparison endpoint), FR-010 (metrics block).

## 4. Frontend (React + Vite + TS)

Single-page, 4-step flow; TanStack Query for upload/segment/poll; HTML canvas for mask.

```
src/
├── api/client.ts            # typed fetch wrapper, base = VITE_API_BASE
├── hooks/                   # useUpload, useSegments, useCreateEdit, useEditPolling
├── components/
│   ├── ImageUploader.tsx        # drag-drop → POST /images
│   ├── RegionSelector/          # drag a rectangle over the image
│   │   └── DragSelect.tsx       # mouse-drag box → [x,y,w,h] in image pixels (primary)
│   ├── PromptPanel.tsx          # preset buttons (Smile/Angry/Sad/Surprised) + free text + params
│   ├── SubmitBar.tsx            # POST /edits, disabled until region+prompt valid
│   └── ComparisonView.tsx       # before/after slider + MetricsPanel
│       └── MetricsPanel.tsx     # CLIP-in-mask, edit success, identity, latency
├── pages/EditorPage.tsx
└── main.tsx
```

### Color theme (monochrome)

Design tokens (define in `frontend/src/styles/theme.css` as CSS custom properties):

| Token | Hex | Role |
|-------|-----|------|
| `--color-bg` | `#111111` | Cod Gray — primary background |
| `--color-fg` | `#FFFFFF` | White — primary text / surfaces on dark |
| `--color-text-secondary` | `#333333` | medium gray — secondary text |
| `--color-text-muted` | `#666666` | medium gray — muted text / borders / disabled |

Rules:
- Default dark UI: `#111111` background, `#FFFFFF` text. Invert for light surfaces
  (cards/inputs): white bg, `#111111` text.
- Secondary/labels `#333333`; placeholders, hints, disabled, hairlines `#666666`.
- No accent/color — buttons, presets, active region chips use grayscale (fill/outline
  contrast). Metrics use weight/icon, not hue.
- Maintain WCAG AA contrast (white-on-#111111 and #111111-on-white both pass).

UI behavior:
- **Upload**: drag-drop or pick → preview, store `image_id`.
- **Region**: drag a rectangle on the image; the dragged box `[x,y,w,h]` (image-pixel
  space) drives the request. Backend turns the box into the edit mask. (Auto-segment
  endpoints remain available but are not the primary UI path.)
- **Prompt**: 4 preset buttons (one-click) or free text; collapsible advanced
  params (strength/steps/seed). Submit disabled until (region|mask) + (prompt|preset).
- **Submit → poll**: POST `/edits` (202), poll `/edits/{id}` until `completed`;
  show spinner + latency.
- **Comparison**: side-by-side / drag-slider of original vs edited + metrics panel.
  Reject → retry with new prompt/region (FR-009).
- Dev: Vite proxy `/api` → `http://backend:8000` (compose) or `localhost:8000`.

## 5. Docker Compose — one-click, hot mount

Root files: `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`,
`.env.example`. `docker compose up` builds + runs both with bind mounts so source
edits hot-reload (uvicorn `--reload`, vite HMR). Shared HF model cache + app-data
volumes persist across restarts. Backend reserves NVIDIA GPU.

- `backend`: bind `./backend:/app`, named volumes `hf-cache:/root/.cache/huggingface`
  + `app-data:/data`, `uvicorn app.infrastructure:app --reload --host 0.0.0.0`,
  GPU via `deploy.resources.reservations.devices`.
- `frontend`: bind `./frontend:/app` + anon volume for `node_modules`, `vite --host`,
  `VITE_API_BASE=/api`, depends_on backend.

(Concrete files written at repo root — see `docker-compose.yml`, `backend/Dockerfile`,
`frontend/Dockerfile`, `.env.example`.)

## 6. Constitution compliance

- Monorepo `backend/` + `frontend/`, coupling only over HTTP (Principle I). ✅
- Backend exactly 4 DDD files, inward deps, pure domain (Principle II). ✅
- Test-first: domain rules + use cases unit-tested, API contract + frontend
  component tests, backend↔frontend integration test (Principle III). ✅
- Python 3.12 + `uv`, typed; React+Vite+ESLint (Principle IV). ✅
- Docker Compose is the single deploy source of truth, env-based config, no
  committed secrets (Principle V). ✅
