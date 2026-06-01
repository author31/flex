# Data Model

**Feature**: 001-facial-expression-editing | **Date**: 2026-06-01

All entities live in `backend/app/domain.py` (pure, no I/O). Persistence is via
`repository.py` Protocols; metadata serialized as JSON sidecars next to image bytes
under the `/data` volume.

## Enums / Value Objects

### FacialRegion (enum)
`FACE | EYES | MOUTH | EYEBROWS | CUSTOM`
- Auto-derivable: FACE, EYES, MOUTH (from anime-face-seg). EYEBROWS = dilated band
  above EYES mask. CUSTOM = client-supplied brush mask.

### ExpressionPreset (enum)
`SMILE | ANGRY | SAD | SURPRISED`
- `preset_to_prompt(preset, region) -> str` maps to an engine prompt.

### EditParams (value object)
| field | type | default | rule |
|-------|------|---------|------|
| strength | float | 0.99 | 0 < strength < 1.0 |
| steps | int | 25 | 10 ≤ steps ≤ 50 |
| guidance | float | 8.0 | 1.0 ≤ guidance ≤ 20.0 |
| seed | int? | null | ≥ 0 if set |

Invalid values → `InvalidEditParams` domain error.

### Mask (value object)
- `data`: binary 2D array (white=edit, black=preserve); `width,height`
- ops: `is_empty()`, `bbox() -> (x,y,w,h)`, `area()`
- Rule: `is_empty()` → `EmptyMask` domain error (FR-002 region required).

## Entities

### CharacterImage
| field | type | notes |
|-------|------|-------|
| id | str (`img_*`) | identity to preserve |
| width / height | int | |
| path | str | bytes location (repository concern) |
| created_at | datetime | |
Rule: image required before segmentation/edit.

### EditJob
| field | type | notes |
|-------|------|-------|
| id | str (`edit_*`) | |
| image_id | str | FK → CharacterImage |
| mask | Mask | resolved region or custom |
| region | FacialRegion | |
| prompt | str | resolved (free text or preset→prompt) |
| params | EditParams | |
| status | `PROCESSING\|COMPLETED\|FAILED` | state machine |
Rules: prompt non-empty (`EmptyPrompt`); exactly one of region/custom-mask;
status transitions PROCESSING→COMPLETED|FAILED only.

### EditResult
| field | type | notes |
|-------|------|-------|
| id | str (`res_*`) | |
| job_id | str | FK → EditJob |
| result_image_id | str (`img_*`) | composited output |
Rule: result image dims == source dims (compositing preserves canvas).

### EvaluationRecord
| field | type | maps to |
|-------|------|---------|
| clip_similarity_in_mask | float | FR-010a, SC-004 |
| edit_success | bool | FR-010b, SC-001 |
| edit_success_score | float | directional CLIP delta |
| identity_preserved | bool | SC-002 (off-mask diff==0 → always true via compositing) |
| latency_ms | int | perf |

## Relationships

```
CharacterImage 1──* EditJob 1──1 EditResult 1──1 EvaluationRecord
                                   │
                          result_image_id → CharacterImage
```

## State transitions (EditJob)

```
(create) → PROCESSING ──success──> COMPLETED
                       └─error───> FAILED
```
Terminal states immutable. Comparison/result endpoints valid only when COMPLETED.
