"""Foundational — SQLAlchemy repositories on in-memory SQLite (no live Postgres)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain import (
    Dataset,
    DatasetItem,
    EditParams,
    EvaluationRecord,
    FacialRegion,
    JobStatus,
    MetricRecord,
    Revision,
    next_revision_number,
)
from app.repository import (
    Base,
    SqlDatasetRepository,
    SqlMetricRecordRepository,
    SqlRevisionRepository,
)


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_dataset_save_get_roundtrip(session_factory):
    repo = SqlDatasetRepository(session_factory)
    ds = Dataset(id="ds_1", name="smiles", model_key="base", params=EditParams(steps=30))
    items = [
        DatasetItem("item_1", "ds_1", "img_1", "smile", 0, box=(0, 0, 10, 10)),
        DatasetItem("item_2", "ds_1", "img_2", "grin", 1, region=FacialRegion.MOUTH),
    ]
    repo.save(ds, items)

    got = repo.get("ds_1")
    assert got is not None and got.params.steps == 30 and got.model_key == "base"
    got_items = repo.get_items("ds_1")
    assert [i.id for i in got_items] == ["item_1", "item_2"]
    assert got_items[0].box == (0, 0, 10, 10)
    assert got_items[1].region is FacialRegion.MOUTH and got_items[1].box is None


def test_dataset_list(session_factory):
    repo = SqlDatasetRepository(session_factory)
    repo.save(Dataset("ds_1", "a", "base", EditParams()), [DatasetItem("i1", "ds_1", "img", "p", 0, box=(0, 0, 1, 1))])
    repo.save(Dataset("ds_2", "b", "base", EditParams()), [DatasetItem("i2", "ds_2", "img", "p", 0, box=(0, 0, 1, 1))])
    assert {d.id for d in repo.list()} == {"ds_1", "ds_2"}


def test_revision_number_monotonic(session_factory):
    repo = SqlRevisionRepository(session_factory)
    assert repo.max_number("ds_1") is None
    n1 = next_revision_number(repo.max_number("ds_1"))
    repo.save(Revision("rev_1", "ds_1", n1, "base", EditParams()))
    assert repo.max_number("ds_1") == 1
    n2 = next_revision_number(repo.max_number("ds_1"))
    repo.save(Revision("rev_2", "ds_1", n2, "finetuned", EditParams()))
    assert repo.max_number("ds_1") == 2
    got = repo.get("ds_1", 2)
    assert got is not None and got.model_key == "finetuned"
    assert {r.number for r in repo.list_for("ds_1")} == {1, 2}


def test_metric_records_roundtrip(session_factory):
    revs = SqlRevisionRepository(session_factory)
    revs.save(Revision("rev_1", "ds_1", 1, "base", EditParams()))
    mrepo = SqlMetricRecordRepository(session_factory)
    ev = EvaluationRecord(0.3, True, 0.1, True, 7)
    mrepo.save(MetricRecord("mr_1", "rev_1", "item_1", JobStatus.COMPLETED, "img_out", ev))
    mrepo.save(MetricRecord("mr_2", "rev_1", "item_2", JobStatus.FAILED, error="boom"))
    recs = {r.dataset_item_id: r for r in mrepo.list_for("rev_1")}
    assert recs["item_1"].evaluation.latency_ms == 7
    assert recs["item_1"].result_image_id == "img_out"
    assert recs["item_2"].status is JobStatus.FAILED and recs["item_2"].error == "boom"


def test_durability_reopen_same_engine(session_factory):
    """SC-006 proxy: data persists across new sessions from the same engine."""
    repo = SqlDatasetRepository(session_factory)
    repo.save(Dataset("ds_x", "n", "base", EditParams()), [DatasetItem("i", "ds_x", "img", "p", 0, box=(0, 0, 1, 1))])
    # Fresh repo instance, same engine → simulates a reconnect.
    assert SqlDatasetRepository(session_factory).get("ds_x") is not None
