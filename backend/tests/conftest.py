"""Test fixtures — fakes for repos and model ports (no torch / no GPU)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.domain import (
    CharacterImage,
    EditJob,
    EditResult,
    EvaluationRecord,
    FacialRegion,
    Mask,
)
from app.infrastructure import Deps, PillowDecoder, Settings, create_app, mask_to_png


# --------------------------- fake repositories --------------------------- #
class FakeImageRepo:
    def __init__(self) -> None:
        self.meta: dict[str, CharacterImage] = {}
        self.blobs: dict[str, bytes] = {}

    def save(self, image: CharacterImage, data: bytes) -> None:
        self.meta[image.id] = image
        self.blobs[image.id] = data

    def get(self, image_id: str) -> CharacterImage | None:
        return self.meta.get(image_id)

    def get_bytes(self, image_id: str) -> bytes | None:
        return self.blobs.get(image_id)


class FakeJobRepo:
    def __init__(self) -> None:
        self.jobs: dict[str, EditJob] = {}

    def save(self, job: EditJob) -> None:
        self.jobs[job.id] = job

    def get(self, job_id: str) -> EditJob | None:
        return self.jobs.get(job_id)


class FakeResultRepo:
    def __init__(self) -> None:
        self.results: dict[str, EditResult] = {}

    def save(self, result: EditResult) -> None:
        self.results[result.job_id] = result

    def get_by_job(self, job_id: str) -> EditResult | None:
        return self.results.get(job_id)


class FakeMaskRepo:
    def __init__(self) -> None:
        self.masks: dict[tuple[str, str], bytes] = {}

    def save(self, image_id: str, region: FacialRegion, png: bytes) -> None:
        self.masks[(image_id, region.value)] = png

    def get(self, image_id: str, region: FacialRegion) -> bytes | None:
        return self.masks.get((image_id, region.value))


# --------------------------- fake model ports --------------------------- #
def _rect_mask(h: int, w: int, box: tuple[int, int, int, int]) -> Mask:
    arr = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = box
    arr[y : y + bh, x : x + bw] = 255
    return Mask(data=arr)


class FakeSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: bytes) -> dict[FacialRegion, Mask]:
        self.calls += 1
        with Image.open(io.BytesIO(image)) as im:
            w, h = im.width, im.height
        from app.infrastructure import derive_eyebrows

        eyes = _rect_mask(h, w, (w // 4, h // 3, w // 2, h // 8))
        return {
            FacialRegion.FACE: _rect_mask(h, w, (w // 8, h // 8, 3 * w // 4, 3 * h // 4)),
            FacialRegion.EYES: eyes,
            FacialRegion.MOUTH: _rect_mask(h, w, (w // 3, 2 * h // 3, w // 3, h // 10)),
            FacialRegion.EYEBROWS: derive_eyebrows(eyes),
        }


class FakeInpainter:
    def __init__(self) -> None:
        self.calls = 0

    def inpaint(self, image, mask, prompt, params) -> bytes:
        self.calls += 1
        # Paint the masked region red on a copy of the original (off-mask preserved).
        orig = Image.open(io.BytesIO(image)).convert("RGB")
        red = Image.new("RGB", orig.size, (255, 0, 0))
        m = Image.fromarray((mask.data > 0).astype(np.uint8) * 255).resize(orig.size)
        out = Image.composite(red, orig, m)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, before, after, mask, prompt) -> EvaluationRecord:
        self.calls += 1
        return EvaluationRecord(
            clip_similarity_in_mask=0.31,
            edit_success=True,
            edit_success_score=0.12,
            identity_preserved=True,
            latency_ms=5,
        )


# --------------------------- composed fixtures --------------------------- #
@pytest.fixture
def fakes() -> dict:
    return {
        "images": FakeImageRepo(),
        "jobs": FakeJobRepo(),
        "results": FakeResultRepo(),
        "masks": FakeMaskRepo(),
        "segmenter": FakeSegmenter(),
        "inpainter": FakeInpainter(),
        "evaluator": FakeEvaluator(),
    }


@pytest.fixture
def deps(fakes) -> Deps:
    return Deps(
        images=fakes["images"],
        jobs=fakes["jobs"],
        results=fakes["results"],
        masks=fakes["masks"],
        segmenter=fakes["segmenter"],
        inpainter=fakes["inpainter"],
        evaluator=fakes["evaluator"],
        decoder=PillowDecoder(),
        settings=Settings(device="cpu", data_dir="/tmp/flex-test"),
    )


@pytest.fixture
def client(deps) -> TestClient:
    return TestClient(create_app(deps))


@pytest.fixture
def png_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), (10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mask_data_url() -> str:
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[40:90, 40:90] = 255
    png = mask_to_png(Mask(data=arr))
    return "data:image/png;base64," + base64.b64encode(png).decode()
