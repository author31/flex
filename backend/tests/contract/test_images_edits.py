"""US1 contract tests — images + edits (custom mask path)."""

from __future__ import annotations


def _upload(client, png_bytes) -> str:
    r = client.post("/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert {"image_id", "width", "height", "url"} <= body.keys()
    return body["image_id"]


def test_upload_and_get_image(client, png_bytes):
    image_id = _upload(client, png_bytes)
    r = client.get(f"/api/v1/images/{image_id}")
    assert r.status_code == 200
    assert r.json()["width"] == 128
    assert client.get(f"/api/v1/images/{image_id}/file").headers["content-type"] == "image/png"


def test_upload_empty_file_422(client):
    r = client.post("/api/v1/images", files={"file": ("c.png", b"", "image/png")})
    assert r.status_code == 422


def test_create_edit_with_custom_mask(client, png_bytes, mask_data_url):
    image_id = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/edits",
        json={"image_id": image_id, "mask": mask_data_url, "prompt": "a wide smile"},
    )
    assert r.status_code == 202, r.text
    edit_id = r.json()["edit_id"]

    # BackgroundTasks run synchronously in TestClient → job already completed.
    s = client.get(f"/api/v1/edits/{edit_id}").json()
    assert s["status"] == "completed"
    assert s["metrics"]["identity_preserved"] is True

    comp = client.get(f"/api/v1/edits/{edit_id}/comparison")
    assert comp.status_code == 200
    body = comp.json()
    assert body["original"]["image_id"] == image_id
    assert body["edited"]["image_id"]
    assert set(body["metrics"]) == {
        "clip_similarity_in_mask",
        "edit_success",
        "edit_success_score",
        "identity_preserved",
        "latency_ms",
    }
    assert client.get(f"/api/v1/edits/{edit_id}/result/file").status_code == 200


def test_create_edit_with_dragged_box(client, png_bytes):
    """Primary UX: select region by dragging a rectangle → backend builds the mask."""
    image_id = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/edits",
        json={"image_id": image_id, "box": [20, 30, 50, 40], "prompt": "a wide smile"},
    )
    assert r.status_code == 202, r.text
    edit_id = r.json()["edit_id"]
    s = client.get(f"/api/v1/edits/{edit_id}").json()
    assert s["status"] == "completed"
    assert s["metrics"]["identity_preserved"] is True


def test_edit_rejects_box_and_mask_together(client, png_bytes, mask_data_url):
    image_id = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/edits",
        json={"image_id": image_id, "box": [0, 0, 10, 10], "mask": mask_data_url, "prompt": "x"},
    )
    assert r.status_code == 422


def test_edit_requires_one_of_prompt_or_preset(client, png_bytes, mask_data_url):
    image_id = _upload(client, png_bytes)
    r = client.post("/api/v1/edits", json={"image_id": image_id, "mask": mask_data_url})
    assert r.status_code == 422


def test_edit_requires_one_of_region_or_mask(client, png_bytes):
    image_id = _upload(client, png_bytes)
    r = client.post("/api/v1/edits", json={"image_id": image_id, "prompt": "smile"})
    assert r.status_code == 422


def test_get_missing_edit_404(client):
    assert client.get("/api/v1/edits/edit_nope").status_code == 404


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert "model_id" in body
