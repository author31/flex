"""US2 application test — SegmentImageUseCase."""

from __future__ import annotations

import pytest

from app.application import SegmentImageUseCase
from app.domain import CharacterImage, FacialRegion
from app.infrastructure import mask_to_png


def _img_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (200, 180, 160)).save(buf, format="PNG")
    return buf.getvalue()


def test_segments_face_eyes_mouth_and_derived_eyebrows(fakes):
    fakes["images"].save(CharacterImage("img_1", 128, 128), _img_bytes())
    use = SegmentImageUseCase(fakes["images"], fakes["masks"], fakes["segmenter"], mask_to_png)

    regions = use.execute("img_1")

    for r in (FacialRegion.FACE, FacialRegion.EYES, FacialRegion.MOUTH, FacialRegion.EYEBROWS):
        assert r in regions
        assert not regions[r].is_empty()
        assert fakes["masks"].get("img_1", r) is not None


def test_eyebrows_band_above_eyes(fakes):
    fakes["images"].save(CharacterImage("img_1", 128, 128), _img_bytes())
    use = SegmentImageUseCase(fakes["images"], fakes["masks"], fakes["segmenter"], mask_to_png)
    regions = use.execute("img_1")
    _, eyes_y, _, _ = regions[FacialRegion.EYES].bbox()
    _, brow_y, _, brow_h = regions[FacialRegion.EYEBROWS].bbox()
    assert brow_y + brow_h <= eyes_y + 1  # band sits above the eyes


def test_segment_unknown_image_raises(fakes):
    use = SegmentImageUseCase(fakes["images"], fakes["masks"], fakes["segmenter"], mask_to_png)
    with pytest.raises(KeyError):
        use.execute("img_missing")
