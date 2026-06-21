#!/usr/bin/env python3
"""LoRA finetune of an SDXL inpainting UNet on a curated manifest.

Consumes the JSONL produced by curate.py (image + box + region + prompt) and
trains a LoRA adapter on the SDXL-inpaint UNet. Each record's box becomes a
rectangle mask exactly like the app (backend box_to_mask); the model learns to
reconstruct the masked region conditioned on the prompt.

NOTE (see specs/001-.../research.md "Reconstruction != Edit"): same-image source
and target means this teaches identity-preserving reconstruction / domain
adaptation, NOT neutral->smile editing. Swap in paired data for true edits.

Requires a CUDA GPU. Reference trainer — tune for your hardware; peft + diffusers
APIs move, so pin versions if a call signature drifts.

Usage:
  python train_inpaint_lora.py curated/manifest.train.jsonl \
      --image-root . --output-dir runs/lora-v1 \
      --train-steps 2000 --batch-size 1 --grad-accum 4 --rank 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VAE_SCALE = 0.13025  # SDXL VAE scaling factor


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def box_to_mask(box: list[int], width: int, height: int) -> np.ndarray:
    """Rectangle mask from [x, y, w, h] — mirrors backend/app box_to_mask."""
    x, y, w, h = box
    arr = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    arr[y0:y1, x0:x1] = 255
    return arr


class InpaintManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        tokenizers: tuple,
        resolution: int,
        caption_dropout: float,
        seed: int,
    ) -> None:
        self.records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        self.image_root = image_root
        self.tok1, self.tok2 = tokenizers
        self.res = resolution
        self.caption_dropout = caption_dropout
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, prompt: str):
        ids1 = self.tok1(
            prompt, padding="max_length", truncation=True,
            max_length=self.tok1.model_max_length, return_tensors="pt",
        ).input_ids[0]
        ids2 = self.tok2(
            prompt, padding="max_length", truncation=True,
            max_length=self.tok2.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return ids1, ids2

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        path = self.image_root / rec["image"]
        img = Image.open(path).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1], HWC

        # scale box from manifest dims to training resolution
        sx, sy = self.res / rec["width"], self.res / rec["height"]
        x, y, w, h = rec["box"]
        box = [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]
        mask = box_to_mask(box, self.res, self.res).astype(np.float32) / 255.0  # HxW {0,1}

        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        mask_t = torch.from_numpy(mask)[None]  # 1xHxW
        masked_pixel_values = pixel_values * (mask_t < 0.5)  # zero the edit region

        prompt = "" if self.rng.random() < self.caption_dropout else rec["prompt"]
        ids1, ids2 = self._tokenize(prompt)
        time_ids = torch.tensor(
            [self.res, self.res, 0, 0, self.res, self.res], dtype=torch.float32
        )
        return {
            "pixel_values": pixel_values,
            "masked_pixel_values": masked_pixel_values,
            "mask": mask_t,
            "input_ids_1": ids1,
            "input_ids_2": ids2,
            "time_ids": time_ids,
        }


# --------------------------------------------------------------------------- #
# Text encoding (SDXL dual encoder)
# --------------------------------------------------------------------------- #
def encode_prompt(text_encoders, ids1, ids2):
    te1, te2 = text_encoders
    out1 = te1(ids1, output_hidden_states=True)
    embeds1 = out1.hidden_states[-2]  # penultimate
    out2 = te2(ids2, output_hidden_states=True)
    pooled = out2[0]  # pooled output (text_embeds)
    embeds2 = out2.hidden_states[-2]
    prompt_embeds = torch.cat([embeds1, embeds2], dim=-1)  # (B, 77, 2048)
    return prompt_embeds, pooled


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="train manifest.jsonl from curate.py")
    ap.add_argument("--image-root", type=Path, default=Path("."), help="base dir for relative image paths")
    ap.add_argument("--base-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--output-dir", type=Path, default=Path("runs/lora"))
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--caption-dropout", type=float, default=0.1, help="empty-prompt prob for CFG")
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    ap.add_argument("--gpu", type=int, default=1, help="CUDA device index (default 1 = second GPU)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        ap.error("CUDA GPU required for SDXL training")
    n_gpu = torch.cuda.device_count()
    if args.gpu >= n_gpu:
        ap.error(f"--gpu {args.gpu} out of range; only {n_gpu} CUDA device(s) visible")

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"==> using GPU {args.gpu}/{n_gpu - 1}: {torch.cuda.get_device_name(args.gpu)}")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.mixed_precision
    ]

    m = args.base_model
    tok1 = AutoTokenizer.from_pretrained(m, subfolder="tokenizer", use_fast=False)
    tok2 = AutoTokenizer.from_pretrained(m, subfolder="tokenizer_2", use_fast=False)
    te1 = CLIPTextModel.from_pretrained(m, subfolder="text_encoder").to(device, weight_dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(m, subfolder="text_encoder_2").to(
        device, weight_dtype
    )
    # VAE stays fp32 for numerical stability.
    vae = AutoencoderKL.from_pretrained(m, subfolder="vae").to(device, torch.float32)
    unet = UNet2DConditionModel.from_pretrained(m, subfolder="unet").to(device, weight_dtype)
    noise_sched = DDPMScheduler.from_pretrained(m, subfolder="scheduler")

    for mod in (te1, te2, vae, unet):
        mod.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    cast_training_params(unet, dtype=torch.float32)  # keep LoRA params fp32
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"==> trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    ds = InpaintManifestDataset(
        args.manifest, args.image_root, (tok1, tok2),
        args.resolution, args.caption_dropout, args.seed,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unet.train()
    step = 0
    accum = 0
    done = False
    epochs = math.ceil(args.train_steps * args.grad_accum / max(1, len(dl)))
    print(f"==> {len(ds)} samples, ~{epochs} epochs to reach {args.train_steps} steps")

    for _ in range(epochs):
        if done:
            break
        for batch in dl:
            with torch.no_grad():
                # latents of the FULL target image
                px = batch["pixel_values"].to(device, torch.float32)
                latents = vae.encode(px).latent_dist.sample() * VAE_SCALE
                # latents of the MASKED image (edit region zeroed)
                mpx = batch["masked_pixel_values"].to(device, torch.float32)
                masked_latents = vae.encode(mpx).latent_dist.sample() * VAE_SCALE
                latents = latents.to(weight_dtype)
                masked_latents = masked_latents.to(weight_dtype)
                # mask down to latent resolution, 1 channel
                mask = F.interpolate(
                    batch["mask"].to(device), size=latents.shape[-2:], mode="nearest"
                ).to(weight_dtype)

                prompt_embeds, pooled = encode_prompt(
                    (te1, te2),
                    batch["input_ids_1"].to(device),
                    batch["input_ids_2"].to(device),
                )
                add_time_ids = batch["time_ids"].to(device, weight_dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_sched.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy = noise_sched.add_noise(latents, noise, timesteps)
            # 9-channel inpaint input: noisy(4) + mask(1) + masked_latents(4)
            model_input = torch.cat([noisy, mask, masked_latents], dim=1)

            pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample
            target = noise  # epsilon objective
            loss = F.mse_loss(pred.float(), target.float()) / args.grad_accum
            loss.backward()
            accum += 1

            if accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                step += 1
                if step % 10 == 0:
                    print(f"  step {step}/{args.train_steps}  loss={loss.item() * args.grad_accum:.4f}")
                if step % args.save_every == 0 or step >= args.train_steps:
                    save_dir = args.output_dir / f"step-{step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    layers = get_peft_model_state_dict(unet)
                    StableDiffusionXLLoraLoaderMixin.save_lora_weights(
                        save_directory=str(save_dir), unet_lora_layers=layers
                    )
                    print(f"  saved LoRA -> {save_dir}")
                if step >= args.train_steps:
                    done = True
                    break

    print(f"==> done. Load in app via DiffusersInpainter pipe.load_lora_weights('{args.output_dir}/step-{args.train_steps}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
