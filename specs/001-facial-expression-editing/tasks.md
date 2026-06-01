---
description: "Task list for Prompt-Guided Facial Expression Editing"
---

# Tasks: Prompt-Guided Facial Expression Editing for Animated Characters

**Input**: Design documents from `specs/001-facial-expression-editing/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/openapi.yaml, research.md, design.md

**Tests**: INCLUDED — constitution Principle III (Test-First) is NON-NEGOTIABLE. Write tests first, confirm fail, then implement.

**Backend layering (constitution Principle II)**: only 4 files —
`backend/app/domain.py`, `repository.py`, `application.py`, `infrastructure.py`.
Many tasks edit the SAME file, so they are sequential (no `[P]`) when sharing a file.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: different file, no incomplete dependency → parallelizable
- **[Story]**: US1 / US2 / US3 (user-story phases only)

---

## Phase 1: Setup (Shared Infrastructure)

- [X] T001 Create backend package skeleton: `backend/app/__init__.py` + empty `backend/app/domain.py`, `repository.py`, `application.py`, `infrastructure.py`
- [X] T002 Create `backend/pyproject.toml` (uv, Python 3.12) with deps: fastapi, uvicorn[standard], pydantic, pydantic-settings, diffusers, transformers, torch, open_clip_torch, pillow, numpy, python-multipart; dev: pytest, httpx, ruff, mypy
- [X] T003 [P] Configure backend lint/format/type: ruff + mypy config in `backend/pyproject.toml`
- [X] T004 [P] Scaffold frontend (React+Vite+TS): `frontend/package.json`, `frontend/vite.config.ts` (dev proxy `/api`→`backend:8000`), `frontend/tsconfig.json`, `frontend/index.html`, `frontend/src/main.tsx`
- [X] T005 [P] Add frontend deps: react, react-dom, @tanstack/react-query, vite, typescript, vitest, @testing-library/react, eslint + config in `frontend/package.json`
- [X] T006 [P] Create monochrome theme tokens in `frontend/src/styles/theme.css` (`--color-bg #111111`, `--color-fg #FFFFFF`, `--color-text-secondary #333333`, `--color-text-muted #666666`) + base reset
- [X] T007 [P] Verify `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`, `.env.example` build; confirm `.env` gitignored

**Checkpoint**: both packages install and boot empty.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ Blocks ALL user stories.**

### Backend domain + persistence (shared by all stories)

- [X] T008 Domain enums + value objects in `backend/app/domain.py`: `FacialRegion`, `ExpressionPreset`, `EditParams` (range validation), `Mask` (is_empty/bbox/area)
- [X] T009 Domain entities + errors in `backend/app/domain.py`: `CharacterImage`, `EditJob` (+status machine), `EditResult`, `EvaluationRecord`; errors `EmptyPrompt`, `EmptyMask`, `InvalidEditParams` (depends T008)
- [X] T010 Persistence Protocols + filesystem impls in `backend/app/repository.py`: `ImageRepository`, `JobRepository`, `ResultRepository` storing bytes + JSON sidecars under `DATA_DIR` (depends T009)
- [X] T011 Application model ports (Protocols) in `backend/app/application.py`: `Segmenter`, `Inpainter`, `Evaluator` (depends T009)
- [X] T012 Infrastructure config + app bootstrap in `backend/app/infrastructure.py`: `Settings` (env), FastAPI app, `/api/v1` router mount, static serving of `DATA_DIR`, exception handlers (domain error→422, missing→404), DI container, startup model warm-load hooks (depends T010, T011)
- [X] T013 [P] `GET /api/v1/health` endpoint in `backend/app/infrastructure.py` returning status/device/model_id/models_ready
- [X] T014 [P] Composition helper (alpha-blend generated region onto original via mask → off-mask byte-identical) in `backend/app/infrastructure.py`

### Backend test harness

- [X] T015 [P] pytest config + fixtures in `backend/tests/conftest.py` (fake repos, fake Segmenter/Inpainter/Evaluator ports, FastAPI TestClient)

### Frontend foundation (shared)

- [X] T016 [P] Typed API client in `frontend/src/api/client.ts` (base `VITE_API_BASE`, typed per `contracts/openapi.yaml`)
- [X] T017 [P] TanStack Query provider + `frontend/src/pages/EditorPage.tsx` shell (4-step layout, theme applied) wired in `frontend/src/main.tsx`
- [X] T018 [P] vitest + testing-library setup in `frontend/src/test/setup.ts` + `frontend/vite.config.ts` test config

**Checkpoint**: domain/persistence/app skeleton ready; health endpoint green; clients can call backend.

---

## Phase 3: User Story 1 - Edit a facial expression from a text prompt (Priority: P1) 🎯 MVP

**Goal**: Upload → mark a custom (brush) mask → free-text prompt → generate edit confined to mask → see before/after + metrics.

**Independent Test**: Upload image, brush the mouth area, prompt "smile", generate; output smiles in region, outside unchanged, metrics returned.

### Tests (write first, MUST fail)

- [X] T019 [P] [US1] Domain unit tests in `backend/tests/unit/test_domain.py`: prompt non-empty, empty mask rejected, EditParams ranges, EditJob status transitions
- [X] T020 [P] [US1] Application test `backend/tests/application/test_generate_edit.py`: GenerateEditUseCase calls inpaint→composite→evaluate, persists result+eval (fake ports)
- [X] T021 [P] [US1] Contract test `backend/tests/contract/test_images_edits.py`: POST /images, POST /edits (custom mask), GET /edits/{id}, GET /edits/{id}/comparison, result file — assert shapes vs openapi
- [X] T022 [P] [US1] Frontend test `frontend/src/components/__tests__/ComparisonView.test.tsx`: renders original/edited + metrics from mocked response

### Implementation

- [X] T023 [US1] `ImportImageUseCase` in `backend/app/application.py` (validate, persist via ImageRepository)
- [X] T024 [US1] `GenerateEditUseCase` in `backend/app/application.py`: resolve custom mask → Inpainter.inpaint → composite → persist EditResult → Evaluator.evaluate → persist EvaluationRecord; async job status (depends T023)
- [X] T025 [US1] `GetEditUseCase` + `GetComparisonUseCase` in `backend/app/application.py` (depends T024)
- [X] T026 [US1] `DiffusersInpainter` adapter in `backend/app/infrastructure.py` (AutoPipelineForInpainting, `MODEL_ID`, strength<1.0/steps/guidance/seed) implementing `Inpainter` port
- [X] T027 [US1] `OpenClipEvaluator` adapter in `backend/app/infrastructure.py` (ViT-L/14; CLIP-in-mask via bbox crop, directional edit-success, identity_preserved=true, latency_ms) implementing `Evaluator` port
- [X] T028 [US1] Image routes in `backend/app/infrastructure.py`: `POST /images`, `GET /images/{id}`, `GET /images/{id}/file` (depends T023)
- [X] T029 [US1] Edit routes in `backend/app/infrastructure.py`: `POST /edits` (custom mask path, 202 + background task), `GET /edits/{id}`, `GET /edits/{id}/comparison`, `GET /edits/{id}/result/file`; one-of region|mask + one-of prompt|preset validation→422 (depends T024, T025)
- [X] T030 [P] [US1] `frontend/src/components/ImageUploader.tsx` (drag-drop → POST /images) + `frontend/src/hooks/useUpload.ts`
- [X] T031 [P] [US1] `frontend/src/components/RegionSelector/BrushTool.tsx` (canvas paint/erase → base64 mask)
- [X] T032 [P] [US1] `frontend/src/components/PromptPanel.tsx` (free-text prompt + advanced params; preset buttons added in US3)
- [X] T033 [P] [US1] `frontend/src/components/SubmitBar.tsx` + `frontend/src/hooks/useCreateEdit.ts` + `frontend/src/hooks/useEditPolling.ts` (POST /edits, poll until completed)
- [X] T034 [P] [US1] `frontend/src/components/ComparisonView.tsx` + `MetricsPanel.tsx` (before/after slider + metrics)
- [X] T035 [US1] Wire US1 flow in `frontend/src/pages/EditorPage.tsx` (upload→brush→prompt→submit→compare) (depends T030–T034)

**Checkpoint**: MVP — end-to-end prompt edit with brush mask works and is testable alone.

---

## Phase 4: User Story 2 - Select and constrain the edit region (Priority: P1)

**Goal**: Auto-segment face into selectable regions (face/eyes/mouth, eyebrows derived); pick a region instead of brushing.

**Independent Test**: Upload, segment, pick "eyes", prompt "surprised eyes", generate; changes contained to eyes.

### Tests (write first, MUST fail)

- [X] T036 [P] [US2] Application test `backend/tests/application/test_segment_image.py`: SegmentImageUseCase returns face/eyes/mouth + derived eyebrows (fake Segmenter)
- [X] T037 [P] [US2] Contract test `backend/tests/contract/test_segments.py`: POST /images/{id}/segments, GET segment file — shapes vs openapi
- [X] T038 [P] [US2] Frontend test `frontend/src/components/__tests__/RegionChips.test.tsx`: renders region chips, selection sets mask

### Implementation

- [X] T039 [US2] `SegmentImageUseCase` in `backend/app/application.py` (call Segmenter; derive eyebrows = dilated band above eye mask; persist region masks)
- [X] T040 [US2] `AnimeFaceSegmenter` adapter in `backend/app/infrastructure.py` (siyeong0 U-Net, `SEGMENTER_WEIGHTS`, 512² → masks face/eyes/mouth) implementing `Segmenter` port
- [X] T041 [US2] Segment routes in `backend/app/infrastructure.py`: `POST /images/{id}/segments`, `GET /images/{id}/segments/{region}/file` (depends T039)
- [X] T042 [US2] Extend `POST /edits` region path in `backend/app/infrastructure.py`: resolve `region`→stored mask (depends T041)
- [X] T043 [P] [US2] `frontend/src/components/RegionSelector/RegionChips.tsx` + `frontend/src/hooks/useSegments.ts` (POST segments, overlay + select)
- [X] T044 [US2] Integrate RegionChips + BrushTool into `frontend/src/components/RegionSelector/index.tsx` and EditorPage (chip OR brush drives mask) (depends T043, T035)

**Checkpoint**: auto-region selection works alongside brush; both US1 and US2 testable.

---

## Phase 5: User Story 3 - Apply preset expression and detail edits (Priority: P2)

**Goal**: One-click presets (smile/angry/sad/surprised) + detail edits (eyes/mouth/eyebrows) instead of typing.

**Independent Test**: Select region, click "angry", generate; angry expression appears.

### Tests (write first, MUST fail)

- [X] T045 [P] [US3] Domain test `backend/tests/unit/test_presets.py`: `preset_to_prompt(preset, region)` maps all 4 presets correctly
- [X] T046 [P] [US3] Contract test `backend/tests/contract/test_presets.py`: GET /presets returns expressions + regions; POST /edits with `preset` resolves to prompt

### Implementation

- [X] T047 [US3] `preset_to_prompt(preset, region)` in `backend/app/domain.py` (4 expressions × region phrasing)
- [X] T048 [US3] `ListPresetsUseCase` in `backend/app/application.py` + resolve `preset`→prompt in GenerateEditUseCase (depends T047)
- [X] T049 [US3] `GET /api/v1/presets` route + preset handling in `POST /edits` in `backend/app/infrastructure.py` (depends T048)
- [X] T050 [P] [US3] Add preset buttons (Smile/Angry/Sad/Surprised) to `frontend/src/components/PromptPanel.tsx` + `frontend/src/hooks/usePresets.ts`

**Checkpoint**: all three stories functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T051 [P] Integration test `backend/tests/contract/test_full_flow.py`: upload→segment→edit(preset)→comparison asserts identity_preserved=true + metrics present
- [X] T052 [P] Frontend integration test `frontend/src/__tests__/editor-flow.test.tsx`: full flow against mocked API
- [X] T053 [P] Backend README `backend/README.md` (layers, run, test) + frontend `frontend/README.md`
- [X] T054 [P] Run ruff+mypy (backend) and eslint (frontend); fix to warning-clean (constitution IV)
- [ ] T055 Validate `quickstart.md` end-to-end via `docker compose up` (GPU); confirm hot reload both packages — **OPEN**: needs NVIDIA GPU + model download, not runnable in this build env. Code/compose/Dockerfiles in place; run on a GPU host to close.
- [X] T056 [P] Error-state UX in `frontend/src/pages/EditorPage.tsx` (422/empty/failed job messaging — FR-008)

---

## Dependencies & Execution Order

- **Setup (P1)** → **Foundational (P2)** → **US1 (P3)** → **US2 (P4)** → **US3 (P5)** → **Polish (P6)**
- Foundational blocks everything. After it:
  - US1 (MVP) independent (uses custom brush mask path).
  - US2 depends on US1 only for shared EditorPage wiring (T044); backend segment work (T036–T042) is independent.
  - US3 depends on Foundational + GenerateEditUseCase (T024); UI preset buttons extend US1 PromptPanel.
- Same-file backend tasks are sequential (domain.py, application.py, infrastructure.py shared).

## Parallel Opportunities

- Setup: T003–T007 in parallel.
- Foundational: T013–T018 parallel after T012.
- US1 tests T019–T022 parallel; US1 frontend T030–T034 parallel.
- US2 tests T036–T038 parallel; US3 tests T045–T046 parallel.
- Polish T051–T054, T056 parallel.

## Implementation Strategy

- **MVP** = Phase 1 + 2 + 3 (US1): full prompt-edit with manual mask. Ship/demo.
- **Increment 2** = US2 auto-region selection.
- **Increment 3** = US3 presets.
- Each story ends at a green, independently testable checkpoint.
