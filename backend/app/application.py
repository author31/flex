"""Application layer — use cases + model ports.

Depends on domain + repository. Model engines (segmenter/inpainter/evaluator) are
ports (Protocols) here; infrastructure implements them. Off-mask identity is
guaranteed by the Inpainter contract (composite outside the mask).
"""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from .domain import (
    AUTO_REGIONS,
    SUPPORTED_PRESETS,
    SUPPORTED_REGIONS,
    CharacterImage,
    EditJob,
    EditParams,
    EditResult,
    EvaluationRecord,
    ExpressionPreset,
    FacialRegion,
    InvalidRequest,
    Mask,
    preset_to_prompt,
    validate_mask,
    validate_prompt,
)
from .repository import (
    ImageRepository,
    JobRepository,
    MaskRepository,
    ResultRepository,
)

# --------------------------------------------------------------------------- #
# Model ports (implemented in infrastructure)
# --------------------------------------------------------------------------- #


class Segmenter(Protocol):
    def segment(self, image: bytes) -> dict[FacialRegion, Mask]:
        """Return masks for face/eyes/mouth (+ derived eyebrows)."""
        ...


class Inpainter(Protocol):
    def inpaint(self, image: bytes, mask: Mask, prompt: str, params: EditParams) -> bytes:
        """Return PNG bytes; pixels OUTSIDE the mask are byte-identical to input."""
        ...


class Evaluator(Protocol):
    def evaluate(
        self, before: bytes, after: bytes, mask: Mask, prompt: str
    ) -> EvaluationRecord: ...


class ImageDecoder(Protocol):
    """Reads image dimensions / decodes mask PNGs (infra-provided)."""

    def dimensions(self, image: bytes) -> tuple[int, int]: ...
    def png_to_mask(self, png: bytes) -> Mask: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Use cases
# --------------------------------------------------------------------------- #


class ImportImageUseCase:
    def __init__(self, images: ImageRepository, decoder: ImageDecoder) -> None:
        self._images = images
        self._decoder = decoder

    def execute(self, data: bytes) -> CharacterImage:
        width, height = self._decoder.dimensions(data)
        image = CharacterImage(id=_new_id("img"), width=width, height=height)
        self._images.save(image, data)
        return image


class SegmentImageUseCase:
    def __init__(
        self,
        images: ImageRepository,
        masks: MaskRepository,
        segmenter: Segmenter,
        mask_to_png,  # callable Mask -> bytes (infra)
    ) -> None:
        self._images = images
        self._masks = masks
        self._segmenter = segmenter
        self._mask_to_png = mask_to_png

    def execute(self, image_id: str) -> dict[FacialRegion, Mask]:
        data = self._images.get_bytes(image_id)
        if data is None:
            raise KeyError(image_id)
        regions = self._segmenter.segment(data)
        for region in AUTO_REGIONS:
            mask = regions.get(region)
            if mask is not None and not mask.is_empty():
                self._masks.save(image_id, region, self._mask_to_png(mask))
        return {r: m for r, m in regions.items() if not m.is_empty()}


class GenerateEditUseCase:
    """Resolves mask → inpaint → persist result → evaluate. Background-task safe."""

    def __init__(
        self,
        images: ImageRepository,
        jobs: JobRepository,
        results: ResultRepository,
        inpainter: Inpainter,
        evaluator: Evaluator,
    ) -> None:
        self._images = images
        self._jobs = jobs
        self._results = results
        self._inpainter = inpainter
        self._evaluator = evaluator

    def create(
        self,
        image_id: str,
        region: FacialRegion,
        prompt: str,
        params: EditParams,
    ) -> EditJob:
        if self._images.get(image_id) is None:
            raise KeyError(image_id)
        job = EditJob(
            id=_new_id("edit"),
            image_id=image_id,
            region=region,
            prompt=validate_prompt(prompt),
            params=params,
        )
        self._jobs.save(job)
        return job

    def run(self, job: EditJob, mask: Mask) -> EditJob:
        """Execute the (slow) generation; called from a background task."""
        try:
            validate_mask(mask)
            original = self._images.get_bytes(job.image_id)
            if original is None:
                raise KeyError(job.image_id)
            edited = self._inpainter.inpaint(original, mask, job.prompt, job.params)

            source = self._images.get(job.image_id)
            assert source is not None
            result_image = CharacterImage(
                id=_new_id("img"), width=source.width, height=source.height
            )
            self._images.save(result_image, edited)
            self._results.save(
                EditResult(id=_new_id("res"), job_id=job.id, result_image_id=result_image.id)
            )

            evaluation = self._evaluator.evaluate(original, edited, mask, job.prompt)
            job.mark_completed(result_image.id, evaluation)
        except Exception as exc:  # noqa: BLE001 — surface failure on the job
            job.mark_failed(str(exc))
        self._jobs.save(job)
        return job


class GetEditUseCase:
    def __init__(self, jobs: JobRepository) -> None:
        self._jobs = jobs

    def execute(self, job_id: str) -> EditJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


class GetComparisonUseCase:
    def __init__(self, jobs: JobRepository) -> None:
        self._jobs = jobs

    def execute(self, job_id: str) -> EditJob:
        return GetEditUseCase(self._jobs).execute(job_id)


class ListPresetsUseCase:
    def execute(self) -> dict[str, list[str]]:
        return {
            "expressions": [p.value for p in SUPPORTED_PRESETS],
            "regions": [r.value for r in SUPPORTED_REGIONS],
        }


# --------------------------------------------------------------------------- #
# Request resolution helpers (pure)
# --------------------------------------------------------------------------- #


def resolve_prompt(prompt: str | None, preset: str | None, region: FacialRegion) -> str:
    """Exactly one of prompt|preset. Preset → engine prompt."""
    if (prompt is None) == (preset is None):
        raise InvalidRequest("provide exactly one of 'prompt' or 'preset'")
    if preset is not None:
        return preset_to_prompt(ExpressionPreset(preset), region)
    return validate_prompt(prompt or "")


def resolve_region(region: str | None, has_custom_selection: bool) -> FacialRegion:
    """Exactly one of a named region OR a custom selection (drag box / brush mask).

    A custom selection (the primary UX: drag a rectangle) always maps to CUSTOM.
    """
    if (region is None) == (not has_custom_selection):
        raise InvalidRequest("provide exactly one of 'region' or a custom selection (box/mask)")
    return FacialRegion(region) if region is not None else FacialRegion.CUSTOM
