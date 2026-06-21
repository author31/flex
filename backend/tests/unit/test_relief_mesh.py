"""Default mesh backend — relief generator produces a valid GLB (no external model)."""

from __future__ import annotations

import io

from PIL import Image

from app.infrastructure import ReliefMeshGenerator, Settings


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 60, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_relief_generates_glb_bytes():
    gen = ReliefMeshGenerator(Settings(device="cpu", data_dir="/tmp/flex-test"))
    data = gen.generate(_png())
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 100
    assert data[:4] == b"glTF"  # GLB magic header
