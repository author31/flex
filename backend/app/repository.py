"""Repository layer — persistence contracts (Protocols) + filesystem implementations.

Depends on domain only. Stores raw bytes + JSON metadata sidecars under DATA_DIR.
Directories are created lazily on first write, so constructing a repository (e.g. at
app import time) never touches the filesystem. Swappable for a database later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .domain import (
    CharacterImage,
    EditJob,
    EditParams,
    EditResult,
    EvaluationRecord,
    FacialRegion,
    JobStatus,
)

# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #


class ImageRepository(Protocol):
    def save(self, image: CharacterImage, data: bytes) -> None: ...
    def get(self, image_id: str) -> CharacterImage | None: ...
    def get_bytes(self, image_id: str) -> bytes | None: ...


class JobRepository(Protocol):
    def save(self, job: EditJob) -> None: ...
    def get(self, job_id: str) -> EditJob | None: ...


class ResultRepository(Protocol):
    def save(self, result: EditResult) -> None: ...
    def get_by_job(self, job_id: str) -> EditResult | None: ...


class MaskRepository(Protocol):
    """Opaque storage of region/segment mask PNG bytes keyed by (image_id, region)."""

    def save(self, image_id: str, region: FacialRegion, png: bytes) -> None: ...
    def get(self, image_id: str, region: FacialRegion) -> bytes | None: ...


# --------------------------------------------------------------------------- #
# Filesystem implementations
# --------------------------------------------------------------------------- #


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


class FileSystemImageRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "images"

    def save(self, image: CharacterImage, data: bytes) -> None:
        d = _ensure(self._dir)
        (d / f"{image.id}.png").write_bytes(data)
        (d / f"{image.id}.json").write_text(
            json.dumps({"id": image.id, "width": image.width, "height": image.height})
        )

    def get(self, image_id: str) -> CharacterImage | None:
        meta = self._dir / f"{image_id}.json"
        if not meta.exists():
            return None
        d = json.loads(meta.read_text())
        return CharacterImage(id=d["id"], width=d["width"], height=d["height"])

    def get_bytes(self, image_id: str) -> bytes | None:
        f = self._dir / f"{image_id}.png"
        return f.read_bytes() if f.exists() else None


class FileSystemJobRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "jobs"

    def save(self, job: EditJob) -> None:
        ev = job.evaluation
        payload = {
            "id": job.id,
            "image_id": job.image_id,
            "region": job.region.value,
            "prompt": job.prompt,
            "params": {
                "strength": job.params.strength,
                "steps": job.params.steps,
                "guidance": job.params.guidance,
                "seed": job.params.seed,
            },
            "status": job.status.value,
            "result_image_id": job.result_image_id,
            "error": job.error,
            "evaluation": (
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
        }
        (_ensure(self._dir) / f"{job.id}.json").write_text(json.dumps(payload))

    def get(self, job_id: str) -> EditJob | None:
        f = self._dir / f"{job_id}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        p = d["params"]
        ev = d.get("evaluation")
        job = EditJob(
            id=d["id"],
            image_id=d["image_id"],
            region=FacialRegion(d["region"]),
            prompt=d["prompt"],
            params=EditParams(
                strength=p["strength"], steps=p["steps"], guidance=p["guidance"], seed=p["seed"]
            ),
            status=JobStatus(d["status"]),
            result_image_id=d.get("result_image_id"),
            error=d.get("error"),
            evaluation=(
                EvaluationRecord(
                    clip_similarity_in_mask=ev["clip_similarity_in_mask"],
                    edit_success=ev["edit_success"],
                    edit_success_score=ev["edit_success_score"],
                    identity_preserved=ev["identity_preserved"],
                    latency_ms=ev["latency_ms"],
                )
                if ev
                else None
            ),
        )
        return job


class FileSystemResultRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "results"

    def save(self, result: EditResult) -> None:
        (_ensure(self._dir) / f"{result.job_id}.json").write_text(
            json.dumps(
                {
                    "id": result.id,
                    "job_id": result.job_id,
                    "result_image_id": result.result_image_id,
                }
            )
        )

    def get_by_job(self, job_id: str) -> EditResult | None:
        f = self._dir / f"{job_id}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return EditResult(id=d["id"], job_id=d["job_id"], result_image_id=d["result_image_id"])


class FileSystemMaskRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "masks"

    def save(self, image_id: str, region: FacialRegion, png: bytes) -> None:
        (_ensure(self._dir) / f"{image_id}_{region.value}.png").write_bytes(png)

    def get(self, image_id: str, region: FacialRegion) -> bytes | None:
        f = self._dir / f"{image_id}_{region.value}.png"
        return f.read_bytes() if f.exists() else None
