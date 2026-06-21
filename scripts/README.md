# scripts/

Utility scripts that live **outside** the backend app — dataset download and
curation for finetuning the inpainting model. They have their own deps
(`requirements.txt`) and are not imported by `backend/app`.

## Setup

```bash
cd scripts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` + `torchvision` must come from the same CUDA build. On a CUDA box install
both from the matching PyTorch index, e.g. for CUDA 13.0:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

torchvision is **required** by the default `gemma4` backend (its image processor
imports `torchvision.transforms.v2`).

## 1. Download the dataset

```bash
./download_dataset.sh ./data/anime-faces
```

Downloads "High-Resolution Anime Face Dataset (512x512)" from Kaggle and unzips
it. If you hit a 403, set `KAGGLE_USERNAME` / `KAGGLE_KEY` (Kaggle → Settings →
Create New Token) and re-run.

## 2. Curate into a finetune manifest

```bash
python curate.py ./data/anime-faces --num-records 5000 --region mouth
```

Each output line in `curated/manifest.jsonl` is one inpainting triple:

```json
{
  "image": "data/anime-faces/00012.png",
  "width": 512, "height": 512,
  "box": [76, 76, 358, 358],
  "region": "mouth",
  "prompt": "a soft closed-mouth smile, focused on the mouth",
  "caption_raw": "a soft closed-mouth smile"
}
```

- **box** is a centered rectangle *smaller* than the image (`--box-scale`,
  default 0.7) — the editable sub-region. The trainer turns it into a mask the
  same way the app does (`box_to_mask`). Ground truth for the masked area = the
  original image pixels.
- **prompt** is formatted like the app's engine prompts
  (`<expression>, <region focus>`), matching `backend/app/domain.py`.

### VLM backends (`--backend`)

| backend     | model (default)                | notes                         |
|-------------|--------------------------------|-------------------------------|
| `gemma4`    | `google/gemma-4-E4B-it`        | default — Gemma 4 multimodal (text+image) |
| `qwen2-vl`  | `Qwen/Qwen2-VL-2B-Instruct`    | alternative open VLM          |
| `blip2`     | `Salesforce/blip2-opt-2.7b`    | lighter, weaker captions      |
| `anthropic` | `claude-opus-4-8`              | API, needs `ANTHROPIC_API_KEY` (uploads images) |

Gemma 4 (released 2026-03-31) is gated on Hugging Face — accept the license and
`huggingface-cli login` before first use. Larger variants:
`google/gemma-4-31B-it`, `google/gemma-4-26B-A4B` (pass via `--model-id`).

Writes three files to `OUT_DIR`: `manifest.jsonl` (all, with a `split` field),
`manifest.train.jsonl`, and `manifest.val.jsonl`.

### Common options

```
--num-records N     max records to emit (default 1000)
--output-dir DIR    default ./curated
--regions LIST      comma list; one picked at random per image
                    (default face,eyes,mouth,eyebrows). Each region gets its own
                    box geometry (eyes upper, mouth lower, etc.)
--box-scale F       face box side as fraction of image, 0<F<1 (default 0.7)
--box-jitter F      center jitter 0..1 of free margin (default 0.3)
--val-frac F        held-out validation fraction (default 0.05)
--recursive         scan input dir recursively
--shuffle --seed N  shuffle before taking --num-records
--device cuda|cpu   transformers backends
--abs-paths         store absolute image paths
```

## 3. Finetune (SDXL inpaint LoRA)

```bash
python train_inpaint_lora.py curated/manifest.train.jsonl \
    --image-root . --output-dir runs/lora-v1 \
    --train-steps 2000 --batch-size 1 --grad-accum 4 --rank 16
```

Trains a LoRA adapter on the SDXL-inpaint UNet. Per record: box → rectangle mask
(same as backend `box_to_mask`), masked image + prompt → reconstruct the region.
Handles SDXL specifics: 9-channel inpaint input (noisy + mask + masked latents),
dual text encoders, micro-conditioning `add_time_ids`, fp32 VAE, 512→1024 box
scaling, caption dropout for CFG. Needs a CUDA GPU.

Load the result back into the app's `DiffusersInpainter`
(`backend/app/infrastructure.py`):

```python
pipe.load_lora_weights("runs/lora-v1/step-2000")
```

> ⚠️ **Reconstruction ≠ Edit.** Source and target are the same image, so this
> teaches identity-preserving reconstruction / anime domain adaptation — **not**
> neutral→smile editing. For true expression edits, train on paired data
> (same identity, different expression). See
> `specs/001-facial-expression-editing/research.md`.

## Caveat — reconstruction vs. edit

This manifest trains the model to **reconstruct** the masked region from context
+ caption (the caption describes the *existing* expression). That teaches
prompt-conditioned region synthesis. For true *expression change* (neutral →
smile) you also need paired data (same identity, different expression); generate
synthetic pairs and add them as a second manifest.
