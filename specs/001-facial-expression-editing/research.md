# Research: Open-Source Models & Stack

**Feature**: 001-facial-expression-editing
**Date**: 2026-06-01
**Status**: grounding for plan.md / design.md

Goal: pick open-source, self-hostable models for the three model jobs — (1) region
masking, (2) prompt-guided inpainting, (3) evaluation — that run behind FastAPI,
ship in Docker, and keep license clean for a commercial prototype.

---

## Job 1 — Region masking (face → region)

| Option | What | Fit | License |
|--------|------|-----|---------|
| **Anime-Face-Segmentation (siyeong0)** ✅ chosen | U-Net (MobileNetV2 encoder), 7 classes: `background, hair, eye, mouth, face, skin, clothes`. Input 3×512×512 → output 7×512×512 | Gives auto masks for **face**, **eyes** (`eye`), **mouth` directly. **Eyebrows has no class** → derived as a band above the `eye` mask, or drawn by manual brush. Anime-trained. | open (GitHub) |
| BiSeNet face-parsing (yakhyo) | Real-time face parsing, ONNX + PyTorch, skin/hair/eyes/etc. Updated 2025-03 | Strong, but tuned for real faces (CelebAMask-HQ); weaker on stylized anime | MIT-style |
| SAM2 (Segment Anything 2) | Promptable point/box → mask | Great for free-form manual selection; no semantic face classes | Apache-2.0 |

**Decision**: Anime-Face-Segmentation for automatic semantic regions. It natively
yields **face**, **eyes**, **mouth**. **Eyebrows** is not a model class, so it is
derived (dilated band above the `eye` mask) or drawn with the manual brush. Manual
brush mask from the frontend is always available as fallback and override. SAM2 is
an optional future "click-to-select" enhancer.

## Job 2 — Prompt-guided inpainting (the editor)

| Option | What | Fit | License caveat |
|--------|------|-----|----------------|
| **SDXL Inpainting 0.1** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) ✅ default | SDXL base + 5 extra UNet channels, 1024², `AutoPipelineForInpainting`, params `strength`(<1.0, ~0.99)/`num_inference_steps`(15–30)/`guidance_scale`(~8). Mask: white=inpaint, black=preserve | diffusers-native, robust, well-documented | **CreativeML OpenRAIL++-M** — commercial use OK |
| **Waifu-Inpaint-XL** (ShinoharaHare) ✅ anime swap | SDXL inpaint from WAI-illustrious (9-channel), anime-tuned | Best fidelity for animated characters; same pipeline interface | check model card before commercial ship |
| FLUX.1 Fill [dev] | BFL fill/inpaint, `FluxFillPipeline`, top quality | Excellent results | **weights non-commercial**; outputs usable commercially. Optional, behind a flag |
| BrushNet / PowerPaint | Plug-in inpaint controllers (SD1.5/SDXL) | Strong prompt-faithfulness; more wiring | varies |

**Decision**: SDXL Inpainting 0.1 as the default engine (clean license, one
`AutoPipelineForInpainting` path). Engine is a swappable port — `MODEL_ID` env var
selects SDXL ↔ Waifu-Inpaint-XL ↔ FLUX Fill without code change. All share the
mask+prompt+image interface.

**Identity guarantee**: regardless of engine, the backend composites the generated
region back onto the original using the mask (alpha-blend at the mask edge only).
Pixels outside the mask are copied byte-for-byte → satisfies SC-002 (100% off-mask
unchanged) deterministically rather than hoping the model leaves them alone.

## Job 3 — Evaluation (the metrics returned to the UI)

| Metric | Method | Source |
|--------|--------|--------|
| `clip_similarity_in_mask` | open_clip (ViT-L/14), crop to mask bbox, cosine(sim) of prompt text vs masked crop | open_clip / CLIP score |
| `edit_success` | directional CLIP: sim(after, target-prompt) − sim(before, target-prompt) > τ; also absolute sim ≥ baseline | GIE-Bench-style grounded editing eval |
| `identity_preserved` | off-mask pixel diff == 0 (guaranteed by compositing) → boolean true | deterministic |
| `latency_ms` | wall-clock of generate step | — |

**Decision**: open_clip ViT-L/14 for both CLIP-in-mask and directional edit-success.
Note (grounded): vanilla CLIP underperforms on masked images, so we crop to the
mask bounding box (not zero-out) before scoring. Baseline τ tuned during impl.

---

## Stack decisions

- **Backend**: Python 3.12, FastAPI, `uv`, `diffusers` + `transformers` + `torch`
  (CUDA), `open_clip_torch`, `pillow`, `numpy`. GPU strongly recommended (SDXL).
- **Frontend**: React + Vite + TypeScript, TanStack Query (job polling), HTML
  canvas for brush/region overlay.
- **Deploy**: Docker Compose, bind-mount both packages for hot reload, shared
  Hugging Face cache volume, NVIDIA GPU passthrough for backend.

## Sources

- [SDXL Inpainting 0.1 (HF)](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) — OpenRAIL++, AutoPipelineForInpainting, 1024²
- [Waifu-Inpaint-XL (HF)](https://huggingface.co/ShinoharaHare/Waifu-Inpaint-XL) — anime SDXL inpaint
- [FLUX.1-Fill-dev (HF)](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) — FluxFillPipeline, non-commercial weights
- [Anime-Face-Segmentation (siyeong0)](https://github.com/siyeong0/Anime-Face-Segmentation) — 7-class anime face U-Net
- [face-parsing / BiSeNet (yakhyo)](https://github.com/yakhyo/face-parsing) — real-face parsing
- [SAM2 (HF docs)](https://huggingface.co/docs/transformers/model_doc/sam2) — promptable masks
- [PowerPaint](https://github.com/open-mmlab/PowerPaint) / [BrushNet](https://tencentarc.github.io/BrushNet/) — inpaint controllers
- [open_clip](https://github.com/mlfoundations/open_clip) — CLIP scoring
- [GIE-Bench (arXiv 2505.11493)](https://arxiv.org/pdf/2505.11493) — grounded edit eval
- [diffusers Inpainting docs](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)
- [Animagine XL 4.0 (HF)](https://huggingface.co/cagliostrolab/animagine-xl-4.0) — anime base reference
