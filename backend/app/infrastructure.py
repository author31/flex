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
from typing import Any, cast

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import application as app_layer
from .application import (
    CreateDatasetUseCase,
    Evaluator,
    GenerateEditUseCase,
    GenerateMeshUseCase,
    GetComparisonUseCase,
    GetDatasetUseCase,
    GetEditUseCase,
    GetMeshUseCase,
    GetRevisionUseCase,
    ImageDecoder,
    ImportImageUseCase,
    Inpainter,
    ListDatasetsUseCase,
    ListModelsUseCase,
    ListPresetsUseCase,
    MeshGenerator,
    ModelProvider,
    RunBatchEditUseCase,
    Segmenter,
    SegmentImageUseCase,
    UpdateDatasetUseCase,
    resolve_prompt,
    resolve_region,
)
from .domain import (
    AUTO_REGIONS,
    DatasetItem,
    DomainError,
    EditJob,
    EditParams,
    EvaluationRecord,
    FacialRegion,
    JobStatus,
    Mask,
    MetricRecord,
    Revision,
    UnknownModel,
)
from .repository import (
    DatasetRepository,
    FileSystemImageRepository,
    FileSystemJobRepository,
    FileSystemMaskRepository,
    FileSystemMeshJobRepository,
    FileSystemMeshRepository,
    FileSystemResultRepository,
    ImageRepository,
    JobRepository,
    MaskRepository,
    MeshJobRepository,
    MeshRepository,
    MetricRecordRepository,
    ResultRepository,
    RevisionRepository,
    SqlDatasetRepository,
    SqlMetricRecordRepository,
    SqlRevisionRepository,
    build_session_factory,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    lora_path: str = ""  # default LoRA for the `base` key (scripts/train_inpaint_lora.py)
    loras: dict[str, str] = {}  # per-checkpoint LoRA dirs, keyed like `models`
    lora_fuse: bool = False  # bake LoRA into weights for inference speed (can't unload after)
    segmenter_weights: str = "/data/models/anime-face-seg.pt"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    mesh_backend: str = "relief"  # "relief" (no model, default) | "triposr"
    mesh_model: str = "stabilityai/TripoSR"
    device: str = "cuda"
    data_dir: str = "/data"
    # Study workspace (feature 002).
    database_url: str = "sqlite:///./flex.db"  # compose overrides with Postgres
    models: dict[str, str] = {}  # named-model registry; empty → {"base": model_id}

    def model_registry(self) -> dict[str, str]:
        """Resolved registry: defaults to a single `base` = the primary inpainter id."""
        return self.models or {"base": self.model_id}

    def lora_registry(self) -> dict[str, str]:
        """Per-checkpoint LoRA dirs, keyed like model_registry(). `lora_path` is the
        back-compat default for the `base` key when `loras` has no `base` entry."""
        reg = dict(self.loras)
        if self.lora_path and "base" not in reg:
            reg["base"] = self.lora_path
        return reg


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
    def __init__(
        self, settings: Settings, model_id: str | None = None, lora_path: str | None = None
    ) -> None:
        self._s = settings
        self._model_id = model_id or settings.model_id  # registry passes a per-key id
        # registry passes a per-checkpoint LoRA; fall back to the global default.
        self._lora_path = lora_path if lora_path is not None else settings.lora_path
        self._pipe: Any = None

    def _load(self) -> Any:
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForInpainting

            dtype = torch.float16 if self._s.device == "cuda" else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(self._model_id, torch_dtype=dtype)
            if self._lora_path:
                # LoRA adapter trained for this exact base by scripts/train_inpaint_lora.py.
                pipe.load_lora_weights(self._lora_path)
                if self._s.lora_fuse:
                    pipe.fuse_lora()
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
        self._model: Any = None

    def _load(self) -> Any:
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
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _load(self) -> Any:
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
        x = self._preprocess(img).unsqueeze(0).to(self._s.device)
        text = self._tokenizer([prompt]).to(self._s.device)
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


class ReliefMeshGenerator(MeshGenerator):
    """Default, dependency-light image→3D: a colored relief (heightmap) mesh.

    Builds a displaced, vertex-colored grid from the image luminance and exports GLB
    via trimesh. No external model / no GPU — works out of the box. For higher-fidelity
    full 3D, set MESH_BACKEND=triposr (requires the TripoSR package + weights).
    """

    _GRID = 96
    _DEPTH = 0.25

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def generate(self, image: bytes) -> bytes:
        import trimesh

        g = self._GRID
        arr = np.asarray(_load_rgb(image).resize((g, g)), dtype=np.float32) / 255.0
        lum = arr.mean(axis=2)

        xs = np.linspace(-1.0, 1.0, g)
        ys = np.linspace(1.0, -1.0, g)
        vx, vy = np.meshgrid(xs, ys)
        vz = (lum - 0.5) * self._DEPTH
        verts = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)

        rgb = (arr.reshape(-1, 3) * 255).astype(np.uint8)
        colors = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, np.uint8)], axis=1)

        idx = np.arange(g * g).reshape(g, g)
        tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces = np.concatenate(
            [np.stack([tl, tr, bl], axis=1), np.stack([tr, br, bl], axis=1)], axis=0
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
        return cast(bytes, mesh.export(file_type="glb"))


class TripoSRMeshGenerator(MeshGenerator):
    """Expert: single-image → 3D mesh via TripoSR. Heavy deps imported lazily.

    Returns GLB bytes. Requires the TripoSR package + weights at runtime (GPU);
    if unavailable the mesh job fails with a clear message rather than crashing import.
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from tsr.system import TSR

            model = TSR.from_pretrained(
                self._s.mesh_model, config_name="config.yaml", weight_name="model.ckpt"
            )
            model.to(self._s.device if torch.cuda.is_available() else "cpu")
            self._model = model
        return self._model

    def generate(self, image: bytes) -> bytes:
        import tempfile

        model = self._load()
        img = _load_rgb(image)
        scene_codes = model([img], device=self._s.device)
        mesh = model.extract_mesh(scene_codes, resolution=256)[0]
        with tempfile.NamedTemporaryFile(suffix=".glb") as tmp:
            mesh.export(tmp.name)  # trimesh → GLB
            with open(tmp.name, "rb") as fh:
                return fh.read()


# --------------------------------------------------------------------------- #
# Dependency container + app factory
# --------------------------------------------------------------------------- #


class InpainterRegistry(ModelProvider):
    """Named-model registry (feature 002): one cached DiffusersInpainter per key."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._registry = settings.model_registry()
        self._loras = settings.lora_registry()
        self._cache: dict[str, Inpainter] = {}

    def models(self) -> dict[str, str]:
        return dict(self._registry)

    def loras(self) -> dict[str, str]:
        return dict(self._loras)

    def default_key(self) -> str:
        return "base" if "base" in self._registry else next(iter(self._registry))

    def inpainter_for(self, key: str) -> Inpainter:
        if key not in self._registry:
            raise UnknownModel(
                f"unknown model key {key!r}; available: {sorted(self._registry)}"
            )
        if key not in self._cache:
            self._cache[key] = DiffusersInpainter(
                self._s, model_id=self._registry[key], lora_path=self._loras.get(key, "")
            )
        return self._cache[key]


@dataclass
class Deps:
    images: ImageRepository
    jobs: JobRepository
    results: ResultRepository
    masks: MaskRepository
    mesh_jobs: MeshJobRepository
    meshes: MeshRepository
    segmenter: Segmenter
    inpainter: Inpainter
    evaluator: Evaluator
    mesh_generator: MeshGenerator
    decoder: ImageDecoder
    settings: Settings
    # Study workspace (feature 002) — optional so existing edit-only tests need not wire them.
    datasets: DatasetRepository | None = None
    revisions: RevisionRepository | None = None
    metric_records: MetricRecordRepository | None = None
    model_provider: ModelProvider | None = None


def default_deps(settings: Settings | None = None) -> Deps:
    s = settings or Settings()
    root = Path(s.data_dir)
    session_factory = build_session_factory(s.database_url)
    return Deps(
        images=FileSystemImageRepository(root),
        jobs=FileSystemJobRepository(root),
        results=FileSystemResultRepository(root),
        masks=FileSystemMaskRepository(root),
        mesh_jobs=FileSystemMeshJobRepository(root),
        meshes=FileSystemMeshRepository(root),
        segmenter=AnimeFaceSegmenter(s),
        inpainter=DiffusersInpainter(s),
        evaluator=OpenClipEvaluator(s),
        mesh_generator=(
            TripoSRMeshGenerator(s) if s.mesh_backend == "triposr" else ReliefMeshGenerator(s)
        ),
        decoder=PillowDecoder(),
        settings=s,
        datasets=SqlDatasetRepository(session_factory),
        revisions=SqlRevisionRepository(session_factory),
        metric_records=SqlMetricRecordRepository(session_factory),
        model_provider=InpainterRegistry(s),
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
    model: str | None = None  # checkpoint key (see GET /api/v1/models); None → default


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


# --- Study workspace schemas (feature 002) ---------------------------------- #


class DatasetItemIn(BaseModel):
    image_id: str
    region: str | None = None
    box: list[int] | None = None
    prompt: str = Field(min_length=1)


class CreateDatasetIn(BaseModel):
    name: str | None = None
    model_key: str
    params: EditParamsIn | None = None
    items: list[DatasetItemIn]


class RunIn(BaseModel):
    model_key: str | None = None  # override the dataset default (used by /compare)


@dataclass
class _Workspace:
    """Non-optional view of the workspace deps (narrowed by `_require_workspace`)."""

    datasets: DatasetRepository
    revisions: RevisionRepository
    metric_records: MetricRecordRepository
    model_provider: ModelProvider


def _params_out(p: EditParams) -> dict:
    return {"strength": p.strength, "steps": p.steps, "guidance": p.guidance, "seed": p.seed}


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

    @api.exception_handler(UnknownModel)
    async def _unknown_model(_: Request, exc: UnknownModel) -> JSONResponse:
        available = list(deps.model_provider.models()) if deps.model_provider else []
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "code": "UnknownModel", "available_models": available},
        )

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

    @api.get("/api/v1/models")
    def list_models() -> dict:
        if deps.model_provider is None:
            raise HTTPException(500, "model registry not configured")
        return ListModelsUseCase(deps.model_provider).execute()

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

        # Select the checkpoint to edit with: requested key, else the default.
        # UnknownModel propagates to the registered handler (422 + available_models).
        if deps.model_provider is not None:
            key = body.model or deps.model_provider.default_key()
            inpainter = deps.model_provider.inpainter_for(key)
        else:
            key = None
            inpainter = deps.inpainter

        gen = GenerateEditUseCase(
            deps.images, deps.jobs, deps.results, inpainter, deps.evaluator
        )
        try:
            job = gen.create(body.image_id, region, prompt, params)
        except KeyError:
            raise HTTPException(404, "image not found") from None
        background.add_task(gen.run, job, mask)
        return {"edit_id": job.id, "status": job.status.value, "model": key}

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

    # --- Study workspace (feature 002): datasets, runs, revisions --------- #

    def _require_workspace() -> _Workspace:
        if (
            deps.datasets is None
            or deps.revisions is None
            or deps.metric_records is None
            or deps.model_provider is None
        ):
            raise HTTPException(500, "workspace persistence not configured")
        return _Workspace(
            datasets=deps.datasets, revisions=deps.revisions,
            metric_records=deps.metric_records, model_provider=deps.model_provider,
        )

    def _params(p: EditParamsIn | None) -> EditParams:
        p = p or EditParamsIn()
        return EditParams(strength=p.strength, steps=p.steps, guidance=p.guidance, seed=p.seed)

    def _item_out(item: DatasetItem) -> dict:
        img = deps.images.get(item.image_id)
        return {
            "id": item.id,
            "image_id": item.image_id,
            "image_url": _image_url(item.image_id),
            "image_width": img.width if img else None,
            "image_height": img.height if img else None,
            "region": item.region.value if item.region else None,
            "box": list(item.box) if item.box else None,
            "prompt": item.prompt,
            "position": item.position,
        }

    def _rev_summary(rev: Revision) -> dict:
        return {
            "number": rev.number,
            "model_key": rev.model_key,
            "status": rev.status.value,
            "created_at": rev.created_at,
        }

    def _resolve_item_mask(item: DatasetItem) -> Mask:
        """Box → rectangle mask; named region → stored (auto-segment if absent)."""
        if item.box is not None:
            img = deps.images.get(item.image_id)
            if img is None:
                raise KeyError(item.image_id)
            return box_to_mask(list(item.box), img.width, img.height)
        region = item.region
        assert region is not None  # a DatasetItem always has a box or a region
        png = deps.masks.get(item.image_id, region)
        if png is None:
            SegmentImageUseCase(
                deps.images, deps.masks, deps.segmenter, mask_to_png
            ).execute(item.image_id)
            png = deps.masks.get(item.image_id, region)
        if png is None:
            raise ValueError(f"no '{region.value}' mask for {item.image_id}")
        return deps.decoder.png_to_mask(png)

    @api.post("/api/v1/datasets", status_code=201)
    def create_dataset(body: CreateDatasetIn) -> dict:
        ws = _require_workspace()
        ds, items = CreateDatasetUseCase(ws.datasets, ws.model_provider).execute(
            body.name, body.model_key, _params(body.params),
            [it.model_dump(exclude_none=True) for it in body.items],
        )
        return {
            "dataset_id": ds.id, "name": ds.name, "model_key": ds.model_key,
            "params": _params_out(ds.params), "items": [_item_out(i) for i in items],
            "revisions": [],
        }

    @api.put("/api/v1/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, body: CreateDatasetIn) -> dict:
        ws = _require_workspace()
        try:
            ds, items = UpdateDatasetUseCase(ws.datasets, ws.model_provider).execute(
                dataset_id, body.name, body.model_key, _params(body.params),
                [it.model_dump(exclude_none=True) for it in body.items],
            )
        except KeyError:
            raise HTTPException(404, "dataset not found") from None
        return {
            "dataset_id": ds.id, "name": ds.name, "model_key": ds.model_key,
            "params": _params_out(ds.params), "items": [_item_out(i) for i in items],
            "revisions": [_rev_summary(r) for r in ws.revisions.list_for(ds.id)],
        }

    @api.get("/api/v1/datasets")
    def list_datasets() -> dict:
        ws = _require_workspace()
        rows = ListDatasetsUseCase(ws.datasets, ws.revisions).execute()
        return {
            "datasets": [
                {
                    "dataset_id": ds.id, "name": ds.name, "model_key": ds.model_key,
                    "item_count": count, "latest_revision": latest,
                }
                for ds, count, latest in rows
            ]
        }

    @api.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        ws = _require_workspace()
        try:
            ds, items, revisions = GetDatasetUseCase(ws.datasets, ws.revisions).execute(dataset_id)
        except KeyError:
            raise HTTPException(404, "dataset not found") from None
        return {
            "dataset_id": ds.id, "name": ds.name, "model_key": ds.model_key,
            "params": _params_out(ds.params), "items": [_item_out(i) for i in items],
            "revisions": [_rev_summary(r) for r in revisions],
        }

    @api.post("/api/v1/datasets/{dataset_id}/runs", status_code=202)
    def run_batch(dataset_id: str, background: BackgroundTasks, body: RunIn | None = None) -> dict:
        ws = _require_workspace()
        uc = RunBatchEditUseCase(
            deps.images, ws.datasets, ws.revisions, ws.metric_records,
            ws.model_provider, deps.evaluator,
        )
        try:
            rev = uc.create(dataset_id, body.model_key if body else None)
        except KeyError:
            raise HTTPException(404, "dataset not found") from None
        background.add_task(uc.run, rev, _resolve_item_mask)
        return {"revision_id": rev.id, "number": rev.number, "status": rev.status.value}

    @api.get("/api/v1/datasets/{dataset_id}/revisions/{number}")
    def get_revision(dataset_id: str, number: int) -> dict:
        ws = _require_workspace()
        try:
            rev, records = GetRevisionUseCase(ws.revisions, ws.metric_records).execute(
                dataset_id, number
            )
        except KeyError:
            raise HTTPException(404, "revision not found") from None
        return {
            "dataset_id": rev.dataset_id, "number": rev.number, "model_key": rev.model_key,
            "params": _params_out(rev.params), "status": rev.status.value,
            "created_at": rev.created_at,
            "records": [_record_out(r) for r in records],
        }

    def _record_out(r: MetricRecord) -> dict:
        ev = r.evaluation
        return {
            "dataset_item_id": r.dataset_item_id,
            "status": r.status.value,
            "result_image_id": r.result_image_id,
            "result_url": _image_url(r.result_image_id) if r.result_image_id else None,
            "metrics": (
                {
                    "clip_similarity_in_mask": ev.clip_similarity_in_mask,
                    "edit_success": ev.edit_success,
                    "edit_success_score": ev.edit_success_score,
                    "identity_preserved": ev.identity_preserved,
                    "latency_ms": ev.latency_ms,
                }
                if ev
                else None
            ),
            "error": r.error,
        }

    # --- Expert: 3D mesh export ------------------------------------------- #

    def _mesh_gen() -> GenerateMeshUseCase:
        return GenerateMeshUseCase(
            deps.images, deps.jobs, deps.mesh_jobs, deps.meshes, deps.mesh_generator
        )

    @api.post("/api/v1/edits/{edit_id}/mesh", status_code=202)
    def create_mesh(edit_id: str, background: BackgroundTasks) -> dict:
        try:
            job = _mesh_gen().create(edit_id)
        except KeyError:
            raise HTTPException(404, "edit not found") from None
        background.add_task(_mesh_gen().run, job)
        return {"mesh_id": job.id, "status": job.status.value}

    @api.get("/api/v1/mesh/{mesh_id}")
    def get_mesh(mesh_id: str) -> dict:
        try:
            job = GetMeshUseCase(deps.mesh_jobs).execute(mesh_id)
        except KeyError:
            raise HTTPException(404, "mesh not found") from None
        url = f"/api/v1/mesh/{job.id}/file" if job.status is JobStatus.COMPLETED else None
        return {"mesh_id": job.id, "status": job.status.value, "url": url, "error": job.error}

    @api.get("/api/v1/mesh/{mesh_id}/file")
    def get_mesh_file(mesh_id: str) -> Response:
        try:
            job = GetMeshUseCase(deps.mesh_jobs).execute(mesh_id)
        except KeyError:
            raise HTTPException(404, "mesh not found") from None
        if job.status is not JobStatus.COMPLETED or not job.mesh_id:
            raise HTTPException(409, f"mesh is {job.status.value}")
        data = deps.meshes.get(job.mesh_id)
        if data is None:
            raise HTTPException(404, "mesh file missing")
        return Response(content=data, media_type="model/gltf-binary")

    return api


# Module-level ASGI app for uvicorn (`app.infrastructure:app`).
app = create_app(default_deps())

__all__ = ["app", "create_app", "default_deps", "Deps", "Settings", "AUTO_REGIONS", "app_layer"]
