# Flex Backend

FastAPI service for prompt-guided facial expression editing. DDD, **exactly four
layers** (constitution Principle II):

| File | Role |
|------|------|
| `app/domain.py` | entities, value objects, enums, rules — pure, no I/O |
| `app/repository.py` | persistence Protocols + filesystem impls (→ domain) |
| `app/application.py` | use cases + model ports (→ domain, repository) |
| `app/infrastructure.py` | FastAPI app, routers, schemas, DI, model adapters |

Model engines are **ports** in `application.py`, implemented as adapters in
`infrastructure.py` (`DiffusersInpainter`, `AnimeFaceSegmenter`, `OpenClipEvaluator`).
Heavy libs (torch/diffusers/open_clip) are imported lazily inside adapters, so the
app and tests import without a GPU.

## Run

```bash
uv sync
uv run uvicorn app.infrastructure:app --reload   # http://localhost:8000/api/v1
```

## Test

```bash
uv run pytest      # unit (domain) + application (fakes) + contract (TestClient)
```

Tests use fakes for repos and model ports — no GPU, no model download.

## Config (env)

`MODEL_ID`, `SEGMENTER_WEIGHTS`, `CLIP_MODEL`, `CLIP_PRETRAINED`, `DEVICE`, `DATA_DIR`.
Swap inpaint engine via `MODEL_ID` (SDXL inpaint default → Waifu-Inpaint-XL → FLUX Fill).

## Identity guarantee

`infrastructure.composite()` keeps original pixels wherever the mask is 0, so off-mask
pixels are byte-identical (SC-002) regardless of the inpainting engine.
