"""Repository layer — persistence contracts (Protocols) + filesystem implementations.

Depends on domain only. Stores raw bytes + JSON metadata sidecars under DATA_DIR.
Directories are created lazily on first write, so constructing a repository (e.g. at
app import time) never touches the filesystem. Swappable for a database later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from sqlalchemy import JSON, Integer, String, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .domain import (
    CharacterImage,
    Dataset,
    DatasetItem,
    EditJob,
    EditParams,
    EditResult,
    EvaluationRecord,
    FacialRegion,
    JobStatus,
    MeshJob,
    MetricRecord,
    Revision,
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


class MeshJobRepository(Protocol):
    def save(self, job: MeshJob) -> None: ...
    def get(self, job_id: str) -> MeshJob | None: ...


class MeshRepository(Protocol):
    """Opaque storage of GLB mesh bytes keyed by mesh id."""

    def save(self, mesh_id: str, data: bytes) -> None: ...
    def get(self, mesh_id: str) -> bytes | None: ...


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


class FileSystemMeshJobRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "mesh_jobs"

    def save(self, job: MeshJob) -> None:
        (_ensure(self._dir) / f"{job.id}.json").write_text(
            json.dumps(
                {
                    "id": job.id,
                    "edit_id": job.edit_id,
                    "source_image_id": job.source_image_id,
                    "status": job.status.value,
                    "mesh_id": job.mesh_id,
                    "error": job.error,
                }
            )
        )

    def get(self, job_id: str) -> MeshJob | None:
        f = self._dir / f"{job_id}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return MeshJob(
            id=d["id"],
            edit_id=d["edit_id"],
            source_image_id=d["source_image_id"],
            status=JobStatus(d["status"]),
            mesh_id=d.get("mesh_id"),
            error=d.get("error"),
        )


class FileSystemMeshRepository:
    def __init__(self, root: Path) -> None:
        self._dir = root / "meshes"

    def save(self, mesh_id: str, data: bytes) -> None:
        (_ensure(self._dir) / f"{mesh_id}.glb").write_bytes(data)

    def get(self, mesh_id: str) -> bytes | None:
        f = self._dir / f"{mesh_id}.glb"
        return f.read_bytes() if f.exists() else None


# --------------------------------------------------------------------------- #
# Study workspace (feature 002) — relational persistence (SQLAlchemy)
#
# Postgres in compose, SQLite in tests. ORM row classes live here (not in domain);
# repositories map rows <-> pure domain dataclasses so domain stays ORM-free.
# Value objects (EditParams, EvaluationRecord) and the box tuple are stored as JSON
# columns so the same schema runs on Postgres and SQLite.
# --------------------------------------------------------------------------- #


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    model_key: Mapped[str] = mapped_column(String)
    params: Mapped[dict] = mapped_column(JSON)


class DatasetItemRow(Base):
    __tablename__ = "dataset_items"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, index=True)
    image_id: Mapped[str] = mapped_column(String)
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    box: Mapped[list | None] = mapped_column(JSON, nullable=True)
    prompt: Mapped[str] = mapped_column(String)
    position: Mapped[int] = mapped_column(Integer)


class RevisionRow(Base):
    __tablename__ = "revisions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, index=True)
    number: Mapped[int] = mapped_column(Integer)
    model_key: Mapped[str] = mapped_column(String)
    params: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[str | None] = mapped_column(String, nullable=True)


class MetricRecordRow(Base):
    __tablename__ = "metric_records"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    revision_id: Mapped[str] = mapped_column(String, index=True)
    dataset_item_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    result_image_id: Mapped[str | None] = mapped_column(String, nullable=True)
    evaluation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)


def _params_to_json(p: EditParams) -> dict:
    return {"strength": p.strength, "steps": p.steps, "guidance": p.guidance, "seed": p.seed}


def _params_from_json(d: dict) -> EditParams:
    return EditParams(
        strength=d["strength"], steps=d["steps"], guidance=d["guidance"], seed=d["seed"]
    )


def _eval_to_json(ev: EvaluationRecord) -> dict:
    return {
        "clip_similarity_in_mask": ev.clip_similarity_in_mask,
        "edit_success": ev.edit_success,
        "edit_success_score": ev.edit_success_score,
        "identity_preserved": ev.identity_preserved,
        "latency_ms": ev.latency_ms,
    }


def _eval_from_json(d: dict | None) -> EvaluationRecord | None:
    if d is None:
        return None
    return EvaluationRecord(
        clip_similarity_in_mask=d["clip_similarity_in_mask"],
        edit_success=d["edit_success"],
        edit_success_score=d["edit_success_score"],
        identity_preserved=d["identity_preserved"],
        latency_ms=d["latency_ms"],
    )


def build_session_factory(url: str) -> sessionmaker[Session]:
    """Engine + sessionmaker; `create_all` so the prototype needs no migrations.

    SQLite (tests) needs StaticPool + a single shared connection so an in-memory DB
    survives across sessions and the background-task threadpool.
    """
    kwargs: dict = {}
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ----------------------------- contracts ----------------------------------- #


class DatasetRepository(Protocol):
    def save(self, dataset: Dataset, items: list[DatasetItem]) -> None: ...
    def get(self, dataset_id: str) -> Dataset | None: ...
    def get_items(self, dataset_id: str) -> list[DatasetItem]: ...
    def delete_items(self, dataset_id: str) -> None: ...
    def list(self) -> list[Dataset]: ...


class RevisionRepository(Protocol):
    def max_number(self, dataset_id: str) -> int | None: ...
    def save(self, revision: Revision) -> None: ...
    def get(self, dataset_id: str, number: int) -> Revision | None: ...
    def list_for(self, dataset_id: str) -> list[Revision]: ...


class MetricRecordRepository(Protocol):
    def save(self, record: MetricRecord) -> None: ...
    def list_for(self, revision_id: str) -> list[MetricRecord]: ...


# ------------------------- SQLAlchemy implementations ---------------------- #


class SqlDatasetRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def save(self, dataset: Dataset, items: list[DatasetItem]) -> None:
        with self._sf.begin() as s:
            s.merge(
                DatasetRow(
                    id=dataset.id,
                    name=dataset.name,
                    model_key=dataset.model_key,
                    params=_params_to_json(dataset.params),
                )
            )
            for it in items:
                s.merge(
                    DatasetItemRow(
                        id=it.id,
                        dataset_id=it.dataset_id,
                        image_id=it.image_id,
                        region=it.region.value if it.region else None,
                        box=list(it.box) if it.box else None,
                        prompt=it.prompt,
                        position=it.position,
                    )
                )

    def get(self, dataset_id: str) -> Dataset | None:
        with self._sf() as s:
            row = s.get(DatasetRow, dataset_id)
            if row is None:
                return None
            return Dataset(
                id=row.id, name=row.name, model_key=row.model_key,
                params=_params_from_json(row.params),
            )

    def delete_items(self, dataset_id: str) -> None:
        with self._sf.begin() as s:
            s.execute(delete(DatasetItemRow).where(DatasetItemRow.dataset_id == dataset_id))

    def get_items(self, dataset_id: str) -> list[DatasetItem]:
        with self._sf() as s:
            rows = s.scalars(
                select(DatasetItemRow)
                .where(DatasetItemRow.dataset_id == dataset_id)
                .order_by(DatasetItemRow.position)
            ).all()
            return [
                DatasetItem(
                    id=r.id, dataset_id=r.dataset_id, image_id=r.image_id, prompt=r.prompt,
                    position=r.position,
                    region=FacialRegion(r.region) if r.region else None,
                    box=tuple(r.box) if r.box else None,
                )
                for r in rows
            ]

    def list(self) -> list[Dataset]:
        with self._sf() as s:
            rows = s.scalars(select(DatasetRow)).all()
            return [
                Dataset(
                    id=r.id, name=r.name, model_key=r.model_key,
                    params=_params_from_json(r.params),
                )
                for r in rows
            ]


class SqlRevisionRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def max_number(self, dataset_id: str) -> int | None:
        with self._sf() as s:
            return s.scalar(
                select(func.max(RevisionRow.number)).where(RevisionRow.dataset_id == dataset_id)
            )

    def save(self, revision: Revision) -> None:
        with self._sf.begin() as s:
            s.merge(
                RevisionRow(
                    id=revision.id,
                    dataset_id=revision.dataset_id,
                    number=revision.number,
                    model_key=revision.model_key,
                    params=_params_to_json(revision.params),
                    status=revision.status.value,
                    created_at=revision.created_at,
                )
            )

    def get(self, dataset_id: str, number: int) -> Revision | None:
        with self._sf() as s:
            row = s.scalars(
                select(RevisionRow).where(
                    RevisionRow.dataset_id == dataset_id, RevisionRow.number == number
                )
            ).first()
            return _revision_from_row(row) if row else None

    def list_for(self, dataset_id: str) -> list[Revision]:
        with self._sf() as s:
            rows = s.scalars(
                select(RevisionRow)
                .where(RevisionRow.dataset_id == dataset_id)
                .order_by(RevisionRow.number)
            ).all()
            return [_revision_from_row(r) for r in rows]


def _revision_from_row(row: RevisionRow) -> Revision:
    return Revision(
        id=row.id, dataset_id=row.dataset_id, number=row.number, model_key=row.model_key,
        params=_params_from_json(row.params), status=JobStatus(row.status),
        created_at=row.created_at,
    )


class SqlMetricRecordRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def save(self, record: MetricRecord) -> None:
        with self._sf.begin() as s:
            s.merge(
                MetricRecordRow(
                    id=record.id,
                    revision_id=record.revision_id,
                    dataset_item_id=record.dataset_item_id,
                    status=record.status.value,
                    result_image_id=record.result_image_id,
                    evaluation=_eval_to_json(record.evaluation) if record.evaluation else None,
                    error=record.error,
                )
            )

    def list_for(self, revision_id: str) -> list[MetricRecord]:
        with self._sf() as s:
            rows = s.scalars(
                select(MetricRecordRow).where(MetricRecordRow.revision_id == revision_id)
            ).all()
            return [
                MetricRecord(
                    id=r.id, revision_id=r.revision_id, dataset_item_id=r.dataset_item_id,
                    status=JobStatus(r.status), result_image_id=r.result_image_id,
                    evaluation=_eval_from_json(r.evaluation), error=r.error,
                )
                for r in rows
            ]
