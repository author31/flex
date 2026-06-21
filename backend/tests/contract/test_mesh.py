"""Expert feature contract tests — 3D mesh export."""

from __future__ import annotations


def _completed_edit(client, png_bytes) -> str:
    image_id = client.post(
        "/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")}
    ).json()["image_id"]
    edit_id = client.post(
        "/api/v1/edits", json={"image_id": image_id, "box": [10, 10, 40, 40], "prompt": "smile"}
    ).json()["edit_id"]
    assert client.get(f"/api/v1/edits/{edit_id}").json()["status"] == "completed"
    return edit_id


def test_export_mesh_flow(client, png_bytes):
    edit_id = _completed_edit(client, png_bytes)

    r = client.post(f"/api/v1/edits/{edit_id}/mesh")
    assert r.status_code == 202, r.text
    mesh_id = r.json()["mesh_id"]

    # BackgroundTasks run synchronously in TestClient → mesh already generated.
    s = client.get(f"/api/v1/mesh/{mesh_id}").json()
    assert s["status"] == "completed"
    assert s["url"] == f"/api/v1/mesh/{mesh_id}/file"

    f = client.get(f"/api/v1/mesh/{mesh_id}/file")
    assert f.status_code == 200
    assert f.headers["content-type"] == "model/gltf-binary"
    assert f.content  # GLB bytes served


def test_mesh_requires_completed_edit(client, png_bytes):
    # An edit id that does not exist → 404
    assert client.post("/api/v1/edits/edit_missing/mesh").status_code == 404


def test_get_missing_mesh_404(client):
    assert client.get("/api/v1/mesh/mesh_nope").status_code == 404
