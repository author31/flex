"""US2/US3 application — RunBatchEdit (numbering, isolation, model override)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.application import CreateDatasetUseCase, GetRevisionUseCase, RunBatchEditUseCase
from app.domain import CharacterImage, EditParams, JobStatus, Mask
from app.repository import (
    SqlDatasetRepository,
    SqlMetricRecordRepository,
    SqlRevisionRepository,
    build_session_factory,
)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _mask(_item) -> Mask:
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    return Mask(data=arr)


def _wire(fakes):
    sf = build_session_factory("sqlite://")
    datasets = SqlDatasetRepository(sf)
    revisions = SqlRevisionRepository(sf)
    records = SqlMetricRecordRepository(sf)
    fakes["images"].save(CharacterImage("img_1", 64, 64), _png())
    fakes["images"].save(CharacterImage("img_2", 64, 64), _png())
    ds, _ = CreateDatasetUseCase(datasets, fakes["model_provider"]).execute(
        "s", "base", EditParams(),
        [
            {"image_id": "img_1", "box": [10, 10, 20, 20], "prompt": "smile"},
            {"image_id": "img_2", "box": [5, 5, 20, 20], "prompt": "grin"},
        ],
    )
    uc = RunBatchEditUseCase(
        fakes["images"], datasets, revisions, records, fakes["model_provider"], fakes["evaluator"]
    )
    return ds, uc, revisions, records


def test_run_numbers_and_records(fakes):
    ds, uc, revisions, records = _wire(fakes)
    rev = uc.create(ds.id, None)
    assert rev.number == 1
    done = uc.run(rev, _mask)
    assert done.status is JobStatus.COMPLETED
    recs = records.list_for(rev.id)
    assert len(recs) == 2
    assert all(r.status is JobStatus.COMPLETED and r.evaluation for r in recs)
    assert fakes["inpainter"].calls == 2

    # Re-run → number 2; prior retained.
    rev2 = uc.create(ds.id, None)
    assert rev2.number == 2
    assert revisions.get(ds.id, 1) is not None


def test_model_override_snapshots_key(fakes):
    ds, uc, revisions, _ = _wire(fakes)
    rev = uc.create(ds.id, "finetuned")
    assert rev.model_key == "finetuned"
    assert revisions.get(ds.id, 1).model_key == "finetuned"


def test_per_item_failure_isolated(fakes):
    ds, uc, revisions, records = _wire(fakes)
    rev = uc.create(ds.id, None)

    # Mask resolver raises for the second item only.
    def flaky(item):
        if item.image_id == "img_2":
            raise ValueError("boom")
        return _mask(item)

    done = uc.run(rev, flaky)
    assert done.status is JobStatus.COMPLETED  # run completes despite one failure
    recs = {r.dataset_item_id: r for r in records.list_for(rev.id)}
    statuses = sorted(r.status.value for r in recs.values())
    assert statuses == ["completed", "failed"]


def test_get_revision_use_case(fakes):
    ds, uc, revisions, records = _wire(fakes)
    rev = uc.run(uc.create(ds.id, None), _mask)
    got, recs = GetRevisionUseCase(revisions, records).execute(ds.id, rev.number)
    assert got.number == 1 and len(recs) == 2
