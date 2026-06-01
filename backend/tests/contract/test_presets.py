"""US3 contract tests — presets list + preset edit path."""

from __future__ import annotations


def test_list_presets(client):
    body = client.get("/api/v1/presets").json()
    assert set(body["expressions"]) == {"smile", "angry", "sad", "surprised"}
    assert "eyes" in body["regions"] and "mouth" in body["regions"]


def test_edit_with_preset(client, png_bytes):
    image_id = client.post(
        "/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")}
    ).json()["image_id"]
    client.post(f"/api/v1/images/{image_id}/segments")
    r = client.post(
        "/api/v1/edits", json={"image_id": image_id, "region": "mouth", "preset": "angry"}
    )
    assert r.status_code == 202, r.text
    edit_id = r.json()["edit_id"]
    comp = client.get(f"/api/v1/edits/{edit_id}/comparison").json()
    assert "angry" in comp["prompt"]
