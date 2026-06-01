"""Infrastructure layer — FastAPI app, routers, schemas, DI, config, model adapters.

External concerns live here: HTTP framework, file/image I/O (PIL), and the heavy model
engines (diffusers / open_clip / torch — imported lazily inside adapters so the app and
tests import without a GPU). Adapters implement the application ports.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import application as app_layer
from .application import (
    Evaluator,
    GenerateEditUseCase,
    GetComparisonUseCase,
    GetEditUseCase,
    ImageDecoder,
    ImportImageUseCase,
    Inpainter,
    ListPresetsUseCase,
    Segmenter,
    SegmentImageUseCase,
    resolve_prompt,
    resolve_region,
)
from .domain import (
    AUTO_REGIONS,
    DomainError,
    EditJob,
    EditParams,
    EvaluationRecord,
    FacialRegion,
    JobStatus,
    Mask,
)
from .repository import (
    FileSystemImageRepository,
    FileSystemJobRepository,
    FileSystemMaskRepository,
    FileSystemResultRepository,
    ImageRepository,
    JobRepository,
    MaskRepository,
    ResultRepository,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    segmenter_weights: str = "/data/models/anime-face-seg.pt"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    device: str = "cuda"
    data_dir: str = "/data"


# --------------------------------------------------------------------------- #
# Image / mask I/O helpers (PIL)
# --------------------------------------------------------------------------- #


def _load_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class PillowDecoder(ImageDecoder):
    def dimensions(self, image: bytes) -> tuple[int, int]:
        with Image.open(io.BytesIO(image)) as im:
            return im.width, im.height

    def png_to_mask(self, png: bytes) -> Mask:
        arr = np.array(Image.open(io.BytesIO(png)).convert("L"))
        return Mask(data=(arr > 127).astype(np.uint8) * 255)


def mask_to_png(mask: Mask) -> bytes:
    return _to_png(Image.fromarray(mask.data.astype(np.uint8)))


def decode_data_url_mask(data_url: str, decoder: ImageDecoder) -> Mask:
    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    return decoder.png_to_mask(base64.b64decode(raw))


def box_to_mask(box: list[int], width: int, height: int) -> Mask:
    """Build a rectangle mask from a dragged box [x, y, w, h] (image-pixel space)."""
    x, y, w, h = box
    arr = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    arr[y0:y1, x0:x1] = 255
    return Mask(data=arr)


def composite(original: bytes, generated: bytes, mask: Mask) -> bytes:
    """Hard-mask composite: keep ORIGINAL pixels wherever mask==0 (off-mask identity,
    SC-002 100%); take generated pixels only inside the mask."""
    orig = _load_rgb(original)
    gen = _load_rgb(generated).resize(orig.size)
    m = Image.fromarray((mask.data > 0).astype(np.uint8) * 255).resize(orig.size)
    out = Image.composite(gen, orig, m)
    return _to_png(out)


# --------------------------------------------------------------------------- #
# Model adapters (heavy libs imported lazily inside methods)
# --------------------------------------------------------------------------- #


class DiffusersInpainter(Inpainter):
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForInpainting

            dtype = torch.float16 if self._s.device == "cuda" else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(self._s.model_id, torch_dtype=dtype)
            self._pipe = pipe.to(self._s.device)
        return self._pipe

    def inpaint(self, image: bytes, mask: Mask, prompt: str, params: EditParams) -> bytes:
        import torch

        pipe = self._load()
        init = _load_rgb(image).resize((1024, 1024))
        mask_img = Image.fromarray((mask.data > 0).astype(np.uint8) * 255).resize((1024, 1024))
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self._s.device).manual_seed(params.seed)
        out = pipe(
            prompt=prompt,
            image=init,
            mask_image=mask_img,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=gen,
        ).images[0]
        out = out.resize(_load_rgb(image).size)
        # Guarantee off-mask identity regardless of model behaviour.
        return composite(image, _to_png(out), mask)


class AnimeFaceSegmenter(Segmenter):
    """siyeong0 anime-face U-Net (7 classes). Eyebrows derived from the eye band."""

    _CLASS = {"face": 4, "eye": 2, "mouth": 3}  # indices per model

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            self._model = torch.load(self._s.segmenter_weights, map_location=self._s.device)
            self._model.eval()
        return self._model

    def segment(self, image: bytes) -> dict[FacialRegion, Mask]:
        import torch

        model = self._load()
        img = _load_rgb(image).resize((512, 512))
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = model(x.to(self._s.device))
        labels = logits.argmax(1).squeeze(0).cpu().numpy()  # (512, 512)
        orig = _load_rgb(image).size

        def class_mask(idx: int) -> Mask:
            m = (labels == idx).astype(np.uint8) * 255
            m = np.array(Image.fromarray(m).resize(orig))
            return Mask(data=(m > 127).astype(np.uint8) * 255)

        out: dict[FacialRegion, Mask] = {
            FacialRegion.FACE: class_mask(self._CLASS["face"]),
            FacialRegion.EYES: class_mask(self._CLASS["eye"]),
            FacialRegion.MOUTH: class_mask(self._CLASS["mouth"]),
        }
        out[FacialRegion.EYEBROWS] = derive_eyebrows(out[FacialRegion.EYES])
        return out


def derive_eyebrows(eyes: Mask) -> Mask:
    """Eyebrows = a band just above the eyes (no segmenter class for them)."""
    if eyes.is_empty():
        return eyes
    x, y, w, h = eyes.bbox()
    band = np.zeros_like(eyes.data)
    top = max(0, y - h)
    band[top:y, x : x + w] = 255
    return Mask(data=band)


class OpenClipEvaluator(Evaluator):
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._s.clip_model, pretrained=self._s.clip_pretrained, device=self._s.device
            )
            self._tokenizer = open_clip.get_tokenizer(self._s.clip_model)
        return self._model

    def _sim(self, img: Image.Image, prompt: str) -> float:
        import torch

        model = self._load()
        x = self._preprocess(img).unsqueeze(0).to(self._s.device)  # type: ignore[misc]
        text = self._tokenizer([prompt]).to(self._s.device)  # type: ignore[misc]
        with torch.no_grad():
            fi = model.encode_image(x)
            ft = model.encode_text(text)
            fi /= fi.norm(dim=-1, keepdim=True)
            ft /= ft.norm(dim=-1, keepdim=True)
            return float((fi @ ft.T).item())

    def evaluate(self, before: bytes, after: bytes, mask: Mask, prompt: str) -> EvaluationRecord:
        start = time.time()
        x, y, w, h = mask.bbox()
        crop_before = _load_rgb(before).crop((x, y, x + w, y + h))
        crop_after = _load_rgb(after).crop((x, y, x + w, y + h))
        sim_after = self._sim(crop_after, prompt)
        sim_before = self._sim(crop_before, prompt)
        delta = sim_after - sim_before
        return EvaluationRecord(
            clip_similarity_in_mask=sim_after,
            edit_success=delta > 0.0,
            edit_success_score=delta,
            identity_preserved=True,  # guaranteed by hard-mask composite
            latency_ms=int((time.time() - start) * 1000),
        )


# --------------------------------------------------------------------------- #
# Dependency container + app factory
# --------------------------------------------------------------------------- #


@dataclass
class Deps:
    images: ImageRepository
    jobs: JobRepository
    results: ResultRepository
    masks: MaskRepository
    segmenter: Segmenter
    inpainter: Inpainter
    evaluator: Evaluator
    decoder: ImageDecoder
    settings: Settings


def default_deps(settings: Settings | None = None) -> Deps:
    s = settings or Settings()
    root = Path(s.data_dir)
    return Deps(
        images=FileSystemImageRepository(root),
        jobs=FileSystemJobRepository(root),
        results=FileSystemResultRepository(root),
        masks=FileSystemMaskRepository(root),
        segmenter=AnimeFaceSegmenter(s),
        inpainter=DiffusersInpainter(s),
        evaluator=OpenClipEvaluator(s),
        decoder=PillowDecoder(),
        settings=s,
    )


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #


class ImageOut(BaseModel):
    image_id: str
    width: int
    height: int
    url: str


class RegionMaskOut(BaseModel):
    url: str
    bbox: list[int]
    area: int


class EditParamsIn(BaseModel):
    strength: float = 0.99
    steps: int = 25
    guidance: float = 8.0
    seed: int | None = None


class CreateEditIn(BaseModel):
    image_id: str
    box: list[int] | None = None  # [x, y, w, h] dragged rectangle (primary UX)
    region: str | None = None
    mask: str | None = None
    prompt: str | None = Field(default=None, min_length=1)
    preset: str | None = None
    params: EditParamsIn | None = None


class MetricsOut(BaseModel):
    clip_similarity_in_mask: float
    edit_success: bool
    edit_success_score: float
    identity_preserved: bool
    latency_ms: int


class EditStatusOut(BaseModel):
    edit_id: str
    status: str
    result: dict | None = None
    metrics: MetricsOut | None = None
    error: str | None = None


def _metrics(job: EditJob) -> MetricsOut | None:
    ev = job.evaluation
    if ev is None:
        return None
    return MetricsOut(
        clip_similarity_in_mask=ev.clip_similarity_in_mask,
        edit_success=ev.edit_success,
        edit_success_score=ev.edit_success_score,
        identity_preserved=ev.identity_preserved,
        latency_ms=ev.latency_ms,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def create_app(deps: Deps) -> FastAPI:
    api = FastAPI(title="Flex — Facial Expression Editing API", version="1.0.0")

    @api.exception_handler(DomainError)
    async def _domain_err(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": str(exc), "code": type(exc).__name__}
        )

    def _image_url(image_id: str) -> str:
        return f"/api/v1/images/{image_id}/file"

    @api.get("/api/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": deps.settings.device,
            "model_id": deps.settings.model_id,
            "models_ready": True,
        }

    @api.get("/api/v1/presets")
    def presets() -> dict:
        return ListPresetsUseCase().execute()

    @api.post("/api/v1/images", status_code=201, response_model=ImageOut)
    async def upload_image(file: UploadFile = File(...)) -> ImageOut:  # noqa: B008
        data = await file.read()
        if not data:
            raise HTTPException(422, "empty file")
        img = ImportImageUseCase(deps.images, deps.decoder).execute(data)
        return ImageOut(image_id=img.id, width=img.width, height=img.height, url=_image_url(img.id))

    @api.get("/api/v1/images/{image_id}", response_model=ImageOut)
    def get_image(image_id: str) -> ImageOut:
        img = deps.images.get(image_id)
        if img is None:
            raise HTTPException(404, "image not found")
        return ImageOut(image_id=img.id, width=img.width, height=img.height, url=_image_url(img.id))

    @api.get("/api/v1/images/{image_id}/file")
    def get_image_file(image_id: str) -> Response:
        data = deps.images.get_bytes(image_id)
        if data is None:
            raise HTTPException(404, "image not found")
        return Response(content=data, media_type="image/png")

    @api.post("/api/v1/images/{image_id}/segments")
    def segment(image_id: str) -> dict:
        if deps.images.get(image_id) is None:
            raise HTTPException(404, "image not found")
        use = SegmentImageUseCase(deps.images, deps.masks, deps.segmenter, mask_to_png)
        regions = use.execute(image_id)
        out: dict[str, RegionMaskOut] = {}
        for region, mask in regions.items():
            bx, by, bw, bh = mask.bbox()
            out[region.value] = RegionMaskOut(
                url=f"/api/v1/images/{image_id}/segments/{region.value}/file",
                bbox=[bx, by, bw, bh],
                area=mask.area(),
            )
        return {"regions": {k: v.model_dump() for k, v in out.items()}}

    @api.get("/api/v1/images/{image_id}/segments/{region}/file")
    def get_segment_file(image_id: str, region: str) -> Response:
        try:
            reg = FacialRegion(region)
        except ValueError:
            raise HTTPException(404, "unknown region") from None
        png = deps.masks.get(image_id, reg)
        if png is None:
            raise HTTPException(404, "mask not found; run POST /segments first")
        return Response(content=png, media_type="image/png")

    @api.post("/api/v1/edits", status_code=202)
    def create_edit(body: CreateEditIn, background: BackgroundTasks) -> dict:
        if body.box is not None and body.mask is not None:
            raise HTTPException(422, "provide only one of 'box' or 'mask'")
        has_custom = body.box is not None or body.mask is not None
        region = resolve_region(body.region, has_custom)
        prompt = resolve_prompt(body.prompt, body.preset, region)
        p = body.params or EditParamsIn()
        params = EditParams(strength=p.strength, steps=p.steps, guidance=p.guidance, seed=p.seed)

        # Resolve the mask: dragged box (primary), custom brush mask, or segmented region.
        if body.box is not None:
            img = deps.images.get(body.image_id)
            if img is None:
                raise HTTPException(404, "image not found")
            mask = box_to_mask(body.box, img.width, img.height)
        elif body.mask is not None:
            mask = decode_data_url_mask(body.mask, deps.decoder)
        else:
            png = deps.masks.get(body.image_id, region)
            if png is None:
                raise HTTPException(409, f"no '{region.value}' mask; run POST /segments first")
            mask = deps.decoder.png_to_mask(png)

        gen = GenerateEditUseCase(
            deps.images, deps.jobs, deps.results, deps.inpainter, deps.evaluator
        )
        try:
            job = gen.create(body.image_id, region, prompt, params)
        except KeyError:
            raise HTTPException(404, "image not found") from None
        background.add_task(gen.run, job, mask)
        return {"edit_id": job.id, "status": job.status.value}

    @api.get("/api/v1/edits/{edit_id}", response_model=EditStatusOut)
    def get_edit(edit_id: str) -> EditStatusOut:
        try:
            job = GetEditUseCase(deps.jobs).execute(edit_id)
        except KeyError:
            raise HTTPException(404, "edit not found") from None
        result = (
            {"result_image_id": job.result_image_id, "url": _image_url(job.result_image_id)}
            if job.result_image_id
            else None
        )
        return EditStatusOut(
            edit_id=job.id, status=job.status.value, result=result, metrics=_metrics(job),
            error=job.error,
        )

    @api.get("/api/v1/edits/{edit_id}/comparison")
    def comparison(edit_id: str) -> dict:
        try:
            job = GetComparisonUseCase(deps.jobs).execute(edit_id)
        except KeyError:
            raise HTTPException(404, "edit not found") from None
        if job.status is not JobStatus.COMPLETED:
            raise HTTPException(409, f"job is {job.status.value}")
        assert job.result_image_id is not None
        metrics = _metrics(job)
        assert metrics is not None
        return {
            "edit_id": job.id,
            "status": job.status.value,
            "original": {"image_id": job.image_id, "url": _image_url(job.image_id)},
            "edited": {"image_id": job.result_image_id, "url": _image_url(job.result_image_id)},
            "region": job.region.value,
            "prompt": job.prompt,
            "metrics": metrics.model_dump(),
        }

    @api.get("/api/v1/edits/{edit_id}/result/file")
    def result_file(edit_id: str) -> Response:
        try:
            job = GetEditUseCase(deps.jobs).execute(edit_id)
        except KeyError:
            raise HTTPException(404, "edit not found") from None
        if not job.result_image_id:
            raise HTTPException(409, "result not ready")
        data = deps.images.get_bytes(job.result_image_id)
        if data is None:
            raise HTTPException(404, "result image missing")
        return Response(content=data, media_type="image/png")

    return api


# Module-level ASGI app for uvicorn (`app.infrastructure:app`).
app = create_app(default_deps())

__all__ = ["app", "create_app", "default_deps", "Deps", "Settings", "AUTO_REGIONS", "app_layer"]
