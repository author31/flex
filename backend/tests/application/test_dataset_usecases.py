"""US1 application — CreateDataset / GetDataset / ListDatasets (SQLite-backed repos)."""

from __future__ import annotations

import pytest

from app.application import (
    CreateDatasetUseCase,
    GetDatasetUseCase,
    ListDatasetsUseCase,
)
from app.domain import EditParams, EmptyDataset, IncompleteItem, UnknownModel
from app.repository import (
    SqlDatasetRepository,
    SqlRevisionRepository,
    build_session_factory,
)


class _Provider:
    def models(self):
        return {"base": "b", "finetuned": "f"}

    def loras(self):
        return {}

    def default_key(self):
        return "base"

    def inpainter_for(self, key):
        raise NotImplementedError


@pytest.fixture
def repos():
    sf = build_session_factory("sqlite://")
    return SqlDatasetRepository(sf), SqlRevisionRepository(sf)


def test_create_persists_dataset_and_items(repos):
    datasets, revisions = repos
    uc = CreateDatasetUseCase(datasets, _Provider())
    ds, items = uc.execute(
        "study", "base", EditParams(steps=20),
        [
            {"image_id": "img_1", "box": [0, 0, 10, 10], "prompt": "smile"},
            {"image_id": "img_2", "region": "eyes", "prompt": "wink"},
        ],
    )
    assert len(items) == 2 and items[0].position == 0
    got, got_items, revs = GetDatasetUseCase(datasets, revisions).execute(ds.id)
    assert got.params.steps == 20
    assert [i.image_id for i in got_items] == ["img_1", "img_2"]
    assert revs == []


def test_create_rejects_unknown_model(repos):
    datasets, _ = repos
    with pytest.raises(UnknownModel):
        CreateDatasetUseCase(datasets, _Provider()).execute(
            None, "ghost", EditParams(), [{"image_id": "i", "box": [0, 0, 1, 1], "prompt": "x"}]
        )


def test_create_rejects_empty_and_incomplete(repos):
    datasets, _ = repos
    uc = CreateDatasetUseCase(datasets, _Provider())
    with pytest.raises(EmptyDataset):
        uc.execute(None, "base", EditParams(), [])
    with pytest.raises(IncompleteItem):
        uc.execute(None, "base", EditParams(), [{"image_id": "i", "prompt": "no region"}])


def test_list_reports_counts(repos):
    datasets, revisions = repos
    uc = CreateDatasetUseCase(datasets, _Provider())
    uc.execute("a", "base", EditParams(), [{"image_id": "i1", "box": [0, 0, 1, 1], "prompt": "p"}])
    uc.execute("b", "base", EditParams(), [
        {"image_id": "i2", "box": [0, 0, 1, 1], "prompt": "p"},
        {"image_id": "i3", "box": [0, 0, 1, 1], "prompt": "p"},
    ])
    listed = {d.name: (count, latest) for d, count, latest in ListDatasetsUseCase(datasets, revisions).execute()}
    assert listed["a"] == (1, None)
    assert listed["b"] == (2, None)
