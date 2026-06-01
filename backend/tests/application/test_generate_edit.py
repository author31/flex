"""US1 application test — GenerateEditUseCase orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from app.application import GenerateEditUseCase
from app.domain import CharacterImage, EditParams, FacialRegion, JobStatus, Mask


def _img_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _mask() -> Mask:
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    return Mask(data=arr)


def _setup(fakes):
    fakes["images"].save(CharacterImage("img_1", 64, 64), _img_bytes())
    return GenerateEditUseCase(
        fakes["images"], fakes["jobs"], fakes["results"], fakes["inpainter"], fakes["evaluator"]
    )


def test_generate_edit_happy_path(fakes):
    gen = _setup(fakes)
    job = gen.create("img_1", FacialRegion.MOUTH, "smile", EditParams())
    assert job.status is JobStatus.PROCESSING

    done = gen.run(job, _mask())
    assert done.status is JobStatus.COMPLETED
    assert fakes["inpainter"].calls == 1
    assert fakes["evaluator"].calls == 1
    assert done.result_image_id is not None
    assert fakes["results"].get_by_job(job.id) is not None
    assert done.evaluation.identity_preserved is True


def test_generate_edit_empty_mask_fails_job(fakes):
    gen = _setup(fakes)
    job = gen.create("img_1", FacialRegion.MOUTH, "smile", EditParams())
    done = gen.run(job, Mask(data=np.zeros((64, 64), dtype=np.uint8)))
    assert done.status is JobStatus.FAILED
    assert fakes["inpainter"].calls == 0


def test_create_unknown_image_raises(fakes):
    gen = _setup(fakes)
    with pytest.raises(KeyError):
        gen.create("img_missing", FacialRegion.MOUTH, "smile", EditParams())
