"""US2 contract tests — segmentation + region edit path."""

from __future__ import annotations


def _upload(client, png_bytes) -> str:
    return client.post(
        "/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")}
    ).json()["image_id"]


def test_segment_returns_regions(client, png_bytes):
    image_id = _upload(client, png_bytes)
    r = client.post(f"/api/v1/images/{image_id}/segments")
    assert r.status_code == 200, r.text
    regions = r.json()["regions"]
    for name in ("face", "eyes", "mouth", "eyebrows"):
        assert name in regions
        assert len(regions[name]["bbox"]) == 4
        assert regions[name]["area"] > 0


def test_segment_mask_file_served(client, png_bytes):
    image_id = _upload(client, png_bytes)
    client.post(f"/api/v1/images/{image_id}/segments")
    r = client.get(f"/api/v1/images/{image_id}/segments/eyes/file")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_edit_with_region_after_segment(client, png_bytes):
    image_id = _upload(client, png_bytes)
    client.post(f"/api/v1/images/{image_id}/segments")
    r = client.post(
        "/api/v1/edits",
        json={"image_id": image_id, "region": "eyes", "prompt": "surprised eyes"},
    )
    assert r.status_code == 202, r.text
    edit_id = r.json()["edit_id"]
    assert client.get(f"/api/v1/edits/{edit_id}").json()["status"] == "completed"


def test_edit_region_without_segment_409(client, png_bytes):
    image_id = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/edits", json={"image_id": image_id, "region": "mouth", "prompt": "smile"}
    )
    assert r.status_code == 409


def test_segment_missing_image_404(client):
    assert client.post("/api/v1/images/img_nope/segments").status_code == 404
