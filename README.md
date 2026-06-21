# Flex

Prompt-guided facial expression editing for anime faces. FastAPI backend (SDXL
inpainting + segmentation + CLIP metrics), React frontend, Postgres for the study
workspace.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Make targets](#make-targets)
- [Notes](#notes)
- [LoRA finetuning](#lora-finetuning)
- [Dev without Docker](#dev-without-docker)

**Package docs**

- [`backend/README.md`](backend/README.md) — FastAPI service, DDD layers, run/test, config
- [`frontend/README.md`](frontend/README.md) — React + Vite app
- [`scripts/README.md`](scripts/README.md) — dataset download, curation, LoRA training

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (backend reserves 1 GPU)

## Quick start

```bash
# 1. (optional) configure secrets/models
cp .env.example .env

# 2. download model checkpoints into Docker volumes, then start the stack
make prepare
make launch
```

Then open:

- Frontend — http://localhost:5000
- API — http://localhost:8000/api/v1
- Health — http://localhost:8000/api/v1/health

First start is slow: backend warm-loads models (~2 min). Check progress with
`make logs`.

## Make targets

| Command | Does |
|---------|------|
| `make build` | build backend + frontend images |
| `make prepare` | build, then download model checkpoints |
| `make launch` | start full stack in background |
| `make logs` | tail service logs |
| `make down` | stop the stack |

## Notes

- **Segmenter weights** are not auto-downloaded (no clean public URL). Grab them
  from [Anime-Face-Segmentation](https://github.com/siyeong0/Anime-Face-Segmentation),
  then run `make prepare SEG_WEIGHTS_URL=<url>`.
- Default inpaint engine is public SDXL — no Hugging Face token needed. Gated
  models (e.g. FLUX Fill) require `HUGGING_FACE_HUB_TOKEN` in `.env`.
- Source is bind-mounted: backend hot-reloads (uvicorn `--reload`), frontend
  uses Vite HMR.

## LoRA finetuning

Scripts in [`scripts/`](scripts/) build a finetune dataset and train a LoRA
adapter for the SDXL inpaint UNet. Separate venv + deps from the backend. Needs a
CUDA GPU.

```bash
cd scripts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch + torchvision from matching CUDA index, e.g. CUDA 13.0:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
# 1. download dataset (Kaggle 512x512 anime faces)
./download_dataset.sh ./data/anime-faces

# 2. caption + curate into a finetune manifest (VLM, default gemma4)
python curate.py ./data/anime-faces --num-records 5000 --region mouth

# 3. train the LoRA adapter
python train_inpaint_lora.py curated/manifest.train.jsonl \
    --image-root . --output-dir runs/lora-v1 \
    --train-steps 2000 --batch-size 1 --grad-accum 4 --rank 16
```

Output adapter (e.g. `runs/lora-v1/step-2000`) is bind-mounted into the backend
at `/data/lora` (see `docker-compose.yml`). Register it via `MODELS` + `LORAS`
env (`.env.example`), pick per-edit through the `model` field of
`POST /api/v1/edits`.

> ⚠️ Default manifest teaches identity-preserving **reconstruction**, not
> neutral→smile **editing**. Real expression edits need paired data. See
> [`scripts/README.md`](scripts/README.md) for full options + caveats.

## Dev without Docker

See [`backend/README.md`](backend/README.md) (uv) and
[`frontend/README.md`](frontend/README.md) (npm) for running each package directly.
