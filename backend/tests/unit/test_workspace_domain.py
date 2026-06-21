"""US foundational — domain rules for the study workspace (feature 002)."""

from __future__ import annotations

import pytest

from app.domain import (
    Dataset,
    DatasetItem,
    EditParams,
    EmptyDataset,
    EmptyPrompt,
    EvaluationRecord,
    FacialRegion,
    IncompleteItem,
    InvalidRequest,
    JobStatus,
    MetricRecord,
    Revision,
    next_revision_number,
    validate_dataset_item,
    validate_dataset_items,
)


# --------------------------- revision numbering --------------------------- #
def test_next_revision_number_from_none_is_one():
    assert next_revision_number(None) == 1


def test_next_revision_number_increments():
    assert next_revision_number(3) == 4


# --------------------------- item validation ------------------------------ #
def test_validate_item_box_ok():
    region, box, prompt = validate_dataset_item(None, [10, 20, 30, 40], "  smile ")
    assert region is None
    assert box == (10, 20, 30, 40)
    assert prompt == "smile"  # trimmed


def test_validate_item_named_region_ok():
    region, box, prompt = validate_dataset_item("mouth", None, "grin")
    assert region is FacialRegion.MOUTH
    assert box is None


def test_validate_item_requires_exactly_one_selection():
    with pytest.raises(InvalidRequest):
        validate_dataset_item("mouth", [1, 2, 3, 4], "x")  # both
    with pytest.raises(InvalidRequest):
        validate_dataset_item(None, None, "x")  # neither


def test_validate_item_blank_prompt_rejected():
    with pytest.raises(EmptyPrompt):
        validate_dataset_item(None, [1, 2, 3, 4], "   ")


def test_validate_item_bad_box_rejected():
    with pytest.raises(InvalidRequest):
        validate_dataset_item(None, [1, 2, 3], "x")  # not 4 ints


# --------------------------- dataset validation --------------------------- #
def test_validate_dataset_items_empty_rejected():
    with pytest.raises(EmptyDataset):
        validate_dataset_items([])


def test_validate_dataset_items_flags_incomplete_row():
    rows = [
        {"image_id": "img_1", "box": [0, 0, 5, 5], "prompt": "ok"},
        {"image_id": "img_2", "prompt": "no selection"},  # missing region+box
    ]
    with pytest.raises(IncompleteItem):
        validate_dataset_items(rows)


# --------------------------- entities round-trip -------------------------- #
def test_dataset_and_items_construct():
    ds = Dataset(id="ds_1", name="smiles", model_key="base", params=EditParams())
    item = DatasetItem(
        id="item_1", dataset_id="ds_1", image_id="img_1",
        region=None, box=(0, 0, 10, 10), prompt="smile", position=0,
    )
    assert ds.model_key == "base"
    assert item.box == (0, 0, 10, 10)


def test_revision_number_unique_contract():
    rev = Revision(
        id="rev_1", dataset_id="ds_1", number=2, model_key="base", params=EditParams(),
    )
    assert rev.status is JobStatus.PROCESSING
    rev.mark_completed()
    assert rev.status is JobStatus.COMPLETED


def test_metric_record_completed_and_failed():
    ev = EvaluationRecord(0.3, True, 0.1, True, 5)
    ok = MetricRecord(
        id="mr_1", revision_id="rev_1", dataset_item_id="item_1",
        status=JobStatus.COMPLETED, result_image_id="img_2", evaluation=ev,
    )
    bad = MetricRecord(
        id="mr_2", revision_id="rev_1", dataset_item_id="item_2",
        status=JobStatus.FAILED, error="boom",
    )
    assert ok.evaluation is ev and ok.result_image_id == "img_2"
    assert bad.error == "boom" and bad.evaluation is None
