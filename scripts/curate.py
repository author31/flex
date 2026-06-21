#!/usr/bin/env python3
"""Curate a face image dataset into an inpainting-finetune manifest.

For each image the script emits one JSONL record holding the inpainting triple
the trainer needs:

  - image  : path to the source image (ground-truth pixels for the masked area)
  - box    : [x, y, w, h] editable region, intentionally SMALLER than the image
             (a face dataset is already a tight crop, so the *editable* sub-region
             — mouth / eyes / face center — is a centered box inside the frame).
             The trainer derives a rectangle mask from this box exactly like the
             app does (see backend box_to_mask).
  - prompt : an expression caption produced by a VLM, formatted to match the
             app's engine prompts ("<expression>, <region focus>").

Usage:
  python curate.py INPUT_DIR --num-records 5000 [options]

VLM backends (--backend):
  gemma4     (default) google/gemma-4-E4B-it via transformers — multimodal Gemma 4
  qwen2-vl   Qwen/Qwen2-VL-2B-Instruct via transformers
  blip2      Salesforce/blip2-opt-2.7b via transformers — lighter, weaker captions
  anthropic  Claude vision via the Anthropic API (needs ANTHROPIC_API_KEY)

Output:
  OUT_DIR/manifest.jsonl   one record per line
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Region-focus phrases mirror backend/app/domain.py (_REGION_FOCUS) so curated
# prompts read the same as the ones the running app sends to the inpainter.
REGION_FOCUS = {
    "face": "on the face",
    "eyes": "focused on the eyes",
    "mouth": "focused on the mouth",
    "eyebrows": "focused on the eyebrows",
    "custom": "in the selected region",
}

VLM_QUESTION = (
    "Describe the character's facial expression in one short phrase suitable as "
    "an image-editing prompt. Focus on the mouth, eyes, and eyebrows. "
    "Answer with the phrase only, no preamble."
)


# --------------------------------------------------------------------------- #
# VLM annotators
# --------------------------------------------------------------------------- #
def _message_text(parsed: object) -> str:
    """Flatten a parsed chat message into plain text.

    Gemma 4's parse_response() returns a message dict (schema-driven), not a
    string. content is either a str or a list of {"type": "text", "text": ...}
    parts; reasoning/thinking parts are skipped.
    """
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        content = parsed.get("content", parsed.get("text", ""))
        return _message_text(content)
    if isinstance(parsed, list):
        parts = [
            p["text"]
            for p in parsed
            if isinstance(p, dict) and p.get("type", "text") == "text" and "text" in p
        ]
        if parts:
            return " ".join(parts)
        return " ".join(_message_text(p) for p in parsed)
    return str(parsed)


class VLMAnnotator(ABC):
    @abstractmethod
    def caption(self, image: Image.Image) -> str: ...


class Gemma4Annotator(VLMAnnotator):
    """Gemma 4 multimodal (text+image) via transformers.

    API per the model card: AutoModelForMultimodalLM + AutoProcessor, image placed
    before text in the chat content, parse_response() to strip Gemma 4's thinking.
    """

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._torch = torch
        self._proc = AutoProcessor.from_pretrained(model_id)
        # device_map="auto" places the model itself; --device is ignored here.
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto"
        )

    def caption(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # image before text
                    {"type": "text", "text": VLM_QUESTION},
                ],
            }
        ]
        inputs = self._proc.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=64)
        response = self._proc.decode(outputs[0][input_len:], skip_special_tokens=False)
        return _message_text(self._proc.parse_response(response)).strip()


class Qwen2VLAnnotator(VLMAnnotator):
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._torch = torch
        self._device = device
        dtype = torch.float16 if device != "cpu" else torch.float32
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self._proc = AutoProcessor.from_pretrained(model_id)

    def caption(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VLM_QUESTION},
                ],
            }
        ]
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._proc(text=[text], images=[image], return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=64)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        out = self._proc.batch_decode(trimmed, skip_special_tokens=True)[0]
        return out.strip()


class Blip2Annotator(VLMAnnotator):
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self._torch = torch
        self._device = device
        dtype = torch.float16 if device != "cpu" else torch.float32
        self._proc = Blip2Processor.from_pretrained(model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)

    def caption(self, image: Image.Image) -> str:
        prompt = "Question: Describe the facial expression. Answer:"
        inputs = self._proc(images=image, text=prompt, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=40)
        out = self._proc.batch_decode(generated, skip_special_tokens=True)[0]
        return out.split("Answer:")[-1].strip()


class AnthropicAnnotator(VLMAnnotator):
    def __init__(self, model_id: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model_id

    def caption(self, image: Image.Image) -> str:
        import io

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=80,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": VLM_QUESTION},
                    ],
                }
            ],
        )
        return msg.content[0].text.strip()


def build_annotator(backend: str, model_id: str | None, device: str) -> VLMAnnotator:
    if backend == "gemma4":
        return Gemma4Annotator(model_id or "google/gemma-4-E4B-it", device)
    if backend == "qwen2-vl":
        return Qwen2VLAnnotator(model_id or "Qwen/Qwen2-VL-2B-Instruct", device)
    if backend == "blip2":
        return Blip2Annotator(model_id or "Salesforce/blip2-opt-2.7b", device)
    if backend == "anthropic":
        return AnthropicAnnotator(model_id or "claude-opus-4-8")
    raise ValueError(f"unknown backend: {backend}")


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    image: str
    width: int
    height: int
    box: list[int]
    region: str
    prompt: str
    caption_raw: str
    split: str


# Per-region box geometry as fractions of the image: (cx, cy, w, h) center+size.
# Tuned for tightly-cropped face portraits. `face` size is overridden by
# --box-scale; the rest are fixed bands. All get jittered (see region_box).
REGION_GEOMETRY: dict[str, tuple[float, float, float, float]] = {
    "face": (0.50, 0.50, 0.70, 0.70),
    "eyes": (0.50, 0.38, 0.60, 0.20),
    "mouth": (0.50, 0.72, 0.42, 0.20),
    "eyebrows": (0.50, 0.27, 0.58, 0.13),
    "custom": (0.50, 0.50, 0.50, 0.50),
}


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    walker = root.rglob("*") if recursive else root.glob("*")
    for p in sorted(walker):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def region_box(
    region: str,
    width: int,
    height: int,
    face_scale: float,
    jitter: float,
    rng: random.Random,
) -> list[int]:
    """A box smaller than the image, placed for `region`, with center jitter.

    jitter is the max center offset as a fraction of the free margin
    (0 = no jitter). Box is always clamped inside the image.
    """
    cx, cy, fw, fh = REGION_GEOMETRY.get(region, REGION_GEOMETRY["custom"])
    if region == "face":
        fw = fh = face_scale
    bw, bh = max(1, int(width * fw)), max(1, int(height * fh))
    free_x, free_y = width - bw, height - bh
    # center position in pixels, then jitter within remaining margin
    base_x = int(cx * width - bw / 2)
    base_y = int(cy * height - bh / 2)
    base_x = max(0, min(free_x, base_x))
    base_y = max(0, min(free_y, base_y))
    off_x = int(rng.uniform(-1, 1) * jitter * min(base_x, free_x - base_x))
    off_y = int(rng.uniform(-1, 1) * jitter * min(base_y, free_y - base_y))
    x = max(0, min(free_x, base_x + off_x))
    y = max(0, min(free_y, base_y + off_y))
    return [x, y, bw, bh]


def to_prompt(caption: str, region: str) -> str:
    cleaned = caption.strip().rstrip(".").strip()
    focus = REGION_FOCUS.get(region, REGION_FOCUS["custom"])
    return f"{cleaned}, {focus}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="directory of face images")
    ap.add_argument("--num-records", type=int, default=1000, help="max records to emit")
    ap.add_argument("--output-dir", type=Path, default=Path("./curated"))
    ap.add_argument(
        "--backend",
        choices=["gemma4", "qwen2-vl", "blip2", "anthropic"],
        default="gemma4",
    )
    ap.add_argument("--model-id", default=None, help="override the VLM model id")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (transformers backends)")
    ap.add_argument(
        "--regions",
        default="face,eyes,mouth,eyebrows",
        help="comma list; one is picked at random per image (default mixes all)",
    )
    ap.add_argument("--box-scale", type=float, default=0.7, help="face box side as fraction of image (<1)")
    ap.add_argument("--box-jitter", type=float, default=0.3, help="center jitter 0..1 of free margin")
    ap.add_argument("--val-frac", type=float, default=0.05, help="fraction held out for validation")
    ap.add_argument("--recursive", action="store_true", help="scan input_dir recursively")
    ap.add_argument("--shuffle", action="store_true", help="shuffle before taking --num-records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abs-paths", action="store_true", help="store absolute image paths in manifest")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        ap.error(f"input_dir not found: {args.input_dir}")
    if not (0.0 < args.box_scale < 1.0):
        ap.error("--box-scale must be in (0, 1)")
    if not (0.0 <= args.val_frac < 1.0):
        ap.error("--val-frac must be in [0, 1)")
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    bad = [r for r in regions if r not in REGION_GEOMETRY]
    if bad:
        ap.error(f"unknown region(s): {bad}; choose from {list(REGION_GEOMETRY)}")

    rng = random.Random(args.seed)
    paths = list(iter_images(args.input_dir, args.recursive))
    if not paths:
        ap.error(f"no images found under {args.input_dir}")
    if args.shuffle:
        rng.shuffle(paths)
    paths = paths[: args.num_records]
    print(f"==> {len(paths)} images selected; loading VLM backend '{args.backend}' ...", file=sys.stderr)

    annotator = build_annotator(args.backend, args.model_id, args.device)

    records: list[Record] = []
    for i, path in enumerate(paths, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 — skip unreadable files, keep going
            print(f"  [skip] {path}: {exc}", file=sys.stderr)
            continue
        w, h = img.size
        region = rng.choice(regions)
        box = region_box(region, w, h, args.box_scale, args.box_jitter, rng)
        try:
            raw = annotator.caption(img)
        except Exception as exc:  # noqa: BLE001 — one bad caption shouldn't kill the run
            print(f"  [skip] {path}: caption failed: {exc}", file=sys.stderr)
            continue
        img_path = str(path.resolve()) if args.abs_paths else str(path)
        records.append(
            Record(
                image=img_path,
                width=w,
                height=h,
                box=box,
                region=region,
                prompt=to_prompt(raw, region),
                caption_raw=raw,
                split="train",  # assigned below
            )
        )
        if i % 50 == 0:
            print(f"  {i}/{len(paths)} processed", file=sys.stderr)

    # Deterministic train/val split.
    rng.shuffle(records)
    n_val = int(len(records) * args.val_frac)
    for r in records[:n_val]:
        r.split = "val"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    train_p = args.output_dir / "manifest.train.jsonl"
    val_p = args.output_dir / "manifest.val.jsonl"
    with manifest.open("w", encoding="utf-8") as fh, train_p.open(
        "w", encoding="utf-8"
    ) as ftr, val_p.open("w", encoding="utf-8") as fva:
        for r in records:
            line = json.dumps(r.__dict__, ensure_ascii=False) + "\n"
            fh.write(line)
            (fva if r.split == "val" else ftr).write(line)

    print(
        f"==> wrote {len(records)} records "
        f"({len(records) - n_val} train / {n_val} val) -> {manifest}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
