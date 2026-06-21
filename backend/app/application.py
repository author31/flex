"""Application layer — use cases + model ports.

Depends on domain + repository. Model engines (segmenter/inpainter/evaluator) are
ports (Protocols) here; infrastructure implements them. Off-mask identity is
guaranteed by the Inpainter contract (composite outside the mask).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from .domain import (
    AUTO_REGIONS,
    SUPPORTED_PRESETS,
    SUPPORTED_REGIONS,
    CharacterImage,
    Dataset,
    DatasetItem,
    EditJob,
    EditParams,
    EditResult,
    EvaluationRecord,
    ExpressionPreset,
    FacialRegion,
    InvalidRequest,
    JobStatus,
    Mask,
    MeshJob,
    MetricRecord,
    Revision,
    UnknownModel,
    next_revision_number,
    preset_to_prompt,
    validate_dataset_item,
    validate_dataset_items,
    validate_mask,
    validate_prompt,
)
from .repository import (
    DatasetRepository,
    ImageRepository,
    JobRepository,
    MaskRepository,
    MeshJobRepository,
    MeshRepository,
    MetricRecordRepository,
    ResultRepository,
    RevisionRepository,
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


class MeshGenerator(Protocol):
    """Expert feature: single-image → 3D mesh. Returns GLB bytes."""

    def generate(self, image: bytes) -> bytes: ...


class ModelProvider(Protocol):
    """Named-model registry (feature 002). Infrastructure implements with diffusers."""

    def models(self) -> dict[str, str]:
        """Map of model key -> underlying model id / weights path."""
        ...

    def loras(self) -> dict[str, str]:
        """Map of model key -> LoRA adapter dir (only keys that have one)."""
        ...

    def default_key(self) -> str: ...

    def inpainter_for(self, key: str) -> Inpainter:
        """Return the Inpainter for `key`; raise UnknownModel if unconfigured."""
        ...


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
        mask_to_png: Callable[[Mask], bytes],  # infra-provided
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


class GenerateMeshUseCase:
    """Expert: export a completed edit's result image to a 3D GLB mesh (async)."""

    def __init__(
        self,
        images: ImageRepository,
        jobs: JobRepository,
        mesh_jobs: MeshJobRepository,
        meshes: MeshRepository,
        generator: MeshGenerator,
    ) -> None:
        self._images = images
        self._jobs = jobs
        self._mesh_jobs = mesh_jobs
        self._meshes = meshes
        self._generator = generator

    def create(self, edit_id: str) -> MeshJob:
        edit = self._jobs.get(edit_id)
        if edit is None:
            raise KeyError(edit_id)
        if edit.status is not JobStatus.COMPLETED or not edit.result_image_id:
            raise InvalidRequest("mesh export requires a completed edit")
        job = MeshJob(
            id=_new_id("mesh"), edit_id=edit_id, source_image_id=edit.result_image_id
        )
        self._mesh_jobs.save(job)
        return job

    def run(self, job: MeshJob) -> MeshJob:
        try:
            img = self._images.get_bytes(job.source_image_id)
            if img is None:
                raise KeyError(job.source_image_id)
            data = self._generator.generate(img)
            mesh_id = _new_id("glb")
            self._meshes.save(mesh_id, data)
            job.mark_completed(mesh_id)
        except Exception as exc:  # noqa: BLE001 — surface failure on the job
            job.mark_failed(str(exc))
        self._mesh_jobs.save(job)
        return job


class GetMeshUseCase:
    def __init__(self, mesh_jobs: MeshJobRepository) -> None:
        self._mesh_jobs = mesh_jobs

    def execute(self, mesh_id: str) -> MeshJob:
        job = self._mesh_jobs.get(mesh_id)
        if job is None:
            raise KeyError(mesh_id)
        return job


class ListPresetsUseCase:
    def execute(self) -> dict[str, list[str]]:
        return {
            "expressions": [p.value for p in SUPPORTED_PRESETS],
            "regions": [r.value for r in SUPPORTED_REGIONS],
        }


class ListModelsUseCase:
    """Feature 002: expose configured model keys for the workspace/compare selectors."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    def execute(self) -> dict:
        loras = self._provider.loras()
        return {
            "models": [
                {"key": k, "model_id": v, "lora": loras.get(k)}
                for k, v in self._provider.models().items()
            ],
            "default": self._provider.default_key(),
        }


# --------------------------------------------------------------------------- #
# Study workspace use cases (feature 002)
# --------------------------------------------------------------------------- #


class CreateDatasetUseCase:
    """Validate rows + chosen model, then persist a dataset and its items (US1)."""

    def __init__(self, datasets: DatasetRepository, provider: ModelProvider) -> None:
        self._datasets = datasets
        self._provider = provider

    def execute(
        self, name: str | None, model_key: str, params: EditParams, items: list[dict]
    ) -> tuple[Dataset, list[DatasetItem]]:
        if model_key not in self._provider.models():
            raise UnknownModel(f"unknown model key {model_key!r}")
        validate_dataset_items(items)  # EmptyDataset / IncompleteItem
        dataset = Dataset(
            id=_new_id("ds"), name=name or "untitled study", model_key=model_key, params=params
        )
        rows: list[DatasetItem] = []
        for pos, row in enumerate(items):
            region, box, prompt = validate_dataset_item(
                row.get("region"), row.get("box"), row.get("prompt") or ""
            )
            rows.append(
                DatasetItem(
                    id=_new_id("item"), dataset_id=dataset.id, image_id=row["image_id"],
                    prompt=prompt, position=pos, region=region, box=box,
                )
            )
        self._datasets.save(dataset, rows)
        return dataset, rows


class UpdateDatasetUseCase:
    """Replace a saved dataset's rows + model/params in place (US1 — edit & re-save)."""

    def __init__(self, datasets: DatasetRepository, provider: ModelProvider) -> None:
        self._datasets = datasets
        self._provider = provider

    def execute(
        self, dataset_id: str, name: str | None, model_key: str, params: EditParams,
        items: list[dict],
    ) -> tuple[Dataset, list[DatasetItem]]:
        existing = self._datasets.get(dataset_id)
        if existing is None:
            raise KeyError(dataset_id)
        if model_key not in self._provider.models():
            raise UnknownModel(f"unknown model key {model_key!r}")
        validate_dataset_items(items)
        dataset = Dataset(
            id=dataset_id, name=name or existing.name, model_key=model_key, params=params
        )
        rows: list[DatasetItem] = []
        for pos, row in enumerate(items):
            region, box, prompt = validate_dataset_item(
                row.get("region"), row.get("box"), row.get("prompt") or ""
            )
            rows.append(
                DatasetItem(
                    id=_new_id("item"), dataset_id=dataset_id, image_id=row["image_id"],
                    prompt=prompt, position=pos, region=region, box=box,
                )
            )
        self._datasets.delete_items(dataset_id)  # replace the row set
        self._datasets.save(dataset, rows)
        return dataset, rows


class GetDatasetUseCase:
    def __init__(self, datasets: DatasetRepository, revisions: RevisionRepository) -> None:
        self._datasets = datasets
        self._revisions = revisions

    def execute(self, dataset_id: str) -> tuple[Dataset, list[DatasetItem], list[Revision]]:
        ds = self._datasets.get(dataset_id)
        if ds is None:
            raise KeyError(dataset_id)
        return ds, self._datasets.get_items(dataset_id), self._revisions.list_for(dataset_id)


class ListDatasetsUseCase:
    def __init__(self, datasets: DatasetRepository, revisions: RevisionRepository) -> None:
        self._datasets = datasets
        self._revisions = revisions

    def execute(self) -> list[tuple[Dataset, int, int | None]]:
        out: list[tuple[Dataset, int, int | None]] = []
        for ds in self._datasets.list():
            items = self._datasets.get_items(ds.id)
            out.append((ds, len(items), self._revisions.max_number(ds.id)))
        return out


class RunBatchEditUseCase:
    """Create a new revision and edit every item under the chosen model (US2).

    `resolve_mask` (infra-provided) turns a DatasetItem into a Mask (box or named region).
    Per-item failures are isolated so the run still completes (FR-011). Off-mask identity
    is guaranteed by the Inpainter contract, so no compositing happens here.
    """

    def __init__(
        self,
        images: ImageRepository,
        datasets: DatasetRepository,
        revisions: RevisionRepository,
        metric_records: MetricRecordRepository,
        provider: ModelProvider,
        evaluator: Evaluator,
    ) -> None:
        self._images = images
        self._datasets = datasets
        self._revisions = revisions
        self._metric_records = metric_records
        self._provider = provider
        self._evaluator = evaluator

    def create(
        self, dataset_id: str, model_key: str | None, created_at: str | None = None
    ) -> Revision:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        if not self._datasets.get_items(dataset_id):
            raise InvalidRequest("cannot run an empty dataset")
        key = model_key or dataset.model_key
        if key not in self._provider.models():
            raise UnknownModel(f"unknown model key {key!r}")
        number = next_revision_number(self._revisions.max_number(dataset_id))
        revision = Revision(
            id=_new_id("rev"), dataset_id=dataset_id, number=number, model_key=key,
            params=dataset.params, created_at=created_at,
        )
        self._revisions.save(revision)
        return revision

    def run(self, revision: Revision, resolve_mask: Callable[[DatasetItem], Mask]) -> Revision:
        items = self._datasets.get_items(revision.dataset_id)
        inpainter = self._provider.inpainter_for(revision.model_key)
        for item in items:
            record = self._run_item(revision, item, inpainter, resolve_mask)
            self._metric_records.save(record)
        revision.mark_completed()
        self._revisions.save(revision)
        return revision

    def _run_item(
        self,
        revision: Revision,
        item: DatasetItem,
        inpainter: Inpainter,
        resolve_mask: Callable[[DatasetItem], Mask],
    ) -> MetricRecord:
        try:
            original = self._images.get_bytes(item.image_id)
            if original is None:
                raise KeyError(item.image_id)
            mask = resolve_mask(item)
            validate_mask(mask)
            edited = inpainter.inpaint(original, mask, item.prompt, revision.params)
            source = self._images.get(item.image_id)
            assert source is not None
            result = CharacterImage(id=_new_id("img"), width=source.width, height=source.height)
            self._images.save(result, edited)
            evaluation = self._evaluator.evaluate(original, edited, mask, item.prompt)
            return MetricRecord(
                id=_new_id("mr"), revision_id=revision.id, dataset_item_id=item.id,
                status=JobStatus.COMPLETED, result_image_id=result.id, evaluation=evaluation,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-item failure
            return MetricRecord(
                id=_new_id("mr"), revision_id=revision.id, dataset_item_id=item.id,
                status=JobStatus.FAILED, error=str(exc),
            )


class GetRevisionUseCase:
    def __init__(
        self, revisions: RevisionRepository, metric_records: MetricRecordRepository
    ) -> None:
        self._revisions = revisions
        self._metric_records = metric_records

    def execute(self, dataset_id: str, number: int) -> tuple[Revision, list[MetricRecord]]:
        revision = self._revisions.get(dataset_id, number)
        if revision is None:
            raise KeyError(f"{dataset_id}#{number}")
        return revision, self._metric_records.list_for(revision.id)


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
