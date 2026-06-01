"""US1 domain unit tests — pure rules, no I/O."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain import (
    EditJob,
    EditParams,
    EmptyMask,
    EmptyPrompt,
    EvaluationRecord,
    FacialRegion,
    InvalidEditParams,
    InvalidTransition,
    JobStatus,
    Mask,
    validate_mask,
    validate_prompt,
)


def _job() -> EditJob:
    return EditJob(
        id="edit_1",
        image_id="img_1",
        region=FacialRegion.MOUTH,
        prompt="smile",
        params=EditParams(),
    )


def _eval() -> EvaluationRecord:
    return EvaluationRecord(0.3, True, 0.1, True, 5)


def test_prompt_must_be_non_empty():
    assert validate_prompt("  smile ") == "smile"
    with pytest.raises(EmptyPrompt):
        validate_prompt("   ")


def test_empty_mask_rejected():
    with pytest.raises(EmptyMask):
        validate_mask(Mask(data=np.zeros((10, 10), dtype=np.uint8)))


def test_mask_bbox_and_area():
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 3:7] = 255
    m = Mask(data=arr)
    assert not m.is_empty()
    assert m.area() == 12
    assert m.bbox() == (3, 2, 4, 3)


@pytest.mark.parametrize(
    "kw",
    [
        {"strength": 1.0},
        {"strength": 0.0},
        {"steps": 5},
        {"steps": 99},
        {"guidance": 0.5},
        {"seed": -1},
    ],
)
def test_edit_params_ranges(kw):
    with pytest.raises(InvalidEditParams):
        EditParams(**kw)


def test_edit_params_defaults_valid():
    p = EditParams()
    assert 0 < p.strength < 1


def test_job_status_transitions():
    job = _job()
    assert job.status is JobStatus.PROCESSING
    job.mark_completed("img_2", _eval())
    assert job.status is JobStatus.COMPLETED
    with pytest.raises(InvalidTransition):
        job.mark_failed("boom")


def test_job_cannot_complete_twice():
    job = _job()
    job.mark_failed("boom")
    assert job.status is JobStatus.FAILED
    with pytest.raises(InvalidTransition):
        job.mark_completed("img_2", _eval())
