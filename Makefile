COMPOSE := docker compose

# Override to fetch the anime-face-segmentation weights (no clean public URL —
# grab from https://github.com/siyeong0/Anime-Face-Segmentation):
#   make prepare SEG_WEIGHTS_URL=<url>
SEG_WEIGHTS_URL ?=

.PHONY: launch prepare build down logs

## launch: start the full stack in the background
launch: prepare
	$(COMPOSE) up -d

## build: build backend + frontend images
build:
	$(COMPOSE) build

## prepare: download all model checkpoints into the local Docker volumes
prepare: build
	@echo "==> SDXL inpainting + open_clip + TripoSR (→ hf-cache volume)"
	$(COMPOSE) run --rm backend python -c "import os; from huggingface_hub import snapshot_download; import open_clip; snapshot_download(os.environ['MODEL_ID']); open_clip.create_model_and_transforms(os.environ.get('CLIP_MODEL','ViT-L-14'), pretrained=os.environ.get('CLIP_PRETRAINED','openai')); snapshot_download(os.environ.get('MESH_MODEL','stabilityai/TripoSR'))"
	@echo "==> anime face segmentation weights (→ app-data volume, /data/models)"
	@if [ -z "$(SEG_WEIGHTS_URL)" ]; then \
		echo "WARN: SEG_WEIGHTS_URL not set — skipping segmenter weights."; \
		echo "      Get them from https://github.com/siyeong0/Anime-Face-Segmentation"; \
		echo "      then re-run: make prepare SEG_WEIGHTS_URL=<url>"; \
	else \
		$(COMPOSE) run --rm backend bash -lc 'mkdir -p /data/models && curl -fL "$(SEG_WEIGHTS_URL)" -o "$$SEGMENTER_WEIGHTS" && echo "saved $$SEGMENTER_WEIGHTS"'; \
	fi
	@echo "==> done. Run: make launch"

## down: stop the stack
down:
	$(COMPOSE) down

## logs: tail service logs
logs:
	$(COMPOSE) logs -f
