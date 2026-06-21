"""Expert feature application test — GenerateMeshUseCase."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.application import GenerateMeshUseCase
from app.domain import (
    CharacterImage,
    EditJob,
    EditParams,
    EvaluationRecord,
    FacialRegion,
    InvalidRequest,
    JobStatus,
)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _completed_edit(fakes) -> EditJob:
    fakes["images"].save(CharacterImage("img_res", 32, 32), _png())
    job = EditJob(
        id="edit_1",
        image_id="img_src",
        region=FacialRegion.CUSTOM,
        prompt="smile",
        params=EditParams(),
    )
    job.mark_completed("img_res", EvaluationRecord(0.3, True, 0.1, True, 5))
    fakes["jobs"].save(job)
    return job


def _uc(fakes) -> GenerateMeshUseCase:
    return GenerateMeshUseCase(
        fakes["images"], fakes["jobs"], fakes["mesh_jobs"], fakes["meshes"], fakes["mesh_generator"]
    )


def test_mesh_happy_path(fakes):
    _completed_edit(fakes)
    uc = _uc(fakes)
    job = uc.create("edit_1")
    assert job.status is JobStatus.PROCESSING

    done = uc.run(job)
    assert done.status is JobStatus.COMPLETED
    assert fakes["mesh_generator"].calls == 1
    assert done.mesh_id is not None
    assert fakes["meshes"].get(done.mesh_id) == b"glTF-FAKE-GLB-BYTES"


def test_mesh_requires_completed_edit(fakes):
    job = EditJob(
        id="edit_2",
        image_id="img_src",
        region=FacialRegion.CUSTOM,
        prompt="smile",
        params=EditParams(),
    )  # still PROCESSING
    fakes["jobs"].save(job)
    with pytest.raises(InvalidRequest):
        _uc(fakes).create("edit_2")


def test_mesh_unknown_edit_raises(fakes):
    with pytest.raises(KeyError):
        _uc(fakes).create("edit_missing")
