"""US2 contract — batch runs + revisions (metrics)."""

from __future__ import annotations


def _upload(client, png_bytes) -> str:
    r = client.post("/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")})
    assert r.status_code == 201
    return r.json()["image_id"]


def _dataset(client, png_bytes, model_key="base") -> str:
    img1, img2 = _upload(client, png_bytes), _upload(client, png_bytes)
    r = client.post(
        "/api/v1/datasets",
        json={
            "name": "s", "model_key": model_key,
            "items": [
                {"image_id": img1, "box": [10, 10, 40, 40], "prompt": "smile"},
                {"image_id": img2, "box": [5, 5, 30, 30], "prompt": "grin"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


def test_run_creates_revision_with_metrics(client, png_bytes):
    ds_id = _dataset(client, png_bytes)
    r = client.post(f"/api/v1/datasets/{ds_id}/runs", json={})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["number"] == 1

    # BackgroundTasks run synchronously in TestClient → revision already completed.
    rev = client.get(f"/api/v1/datasets/{ds_id}/revisions/1").json()
    assert rev["status"] == "completed"
    assert rev["model_key"] == "base"
    assert len(rev["records"]) == 2
    for rec in rev["records"]:
        assert rec["status"] == "completed"
        assert set(rec["metrics"]) == {
            "clip_similarity_in_mask", "edit_success", "edit_success_score",
            "identity_preserved", "latency_ms",
        }
        assert rec["result_url"]


def test_rerun_increments_revision_number(client, png_bytes):
    ds_id = _dataset(client, png_bytes)
    assert client.post(f"/api/v1/datasets/{ds_id}/runs", json={}).json()["number"] == 1
    assert client.post(f"/api/v1/datasets/{ds_id}/runs", json={}).json()["number"] == 2
    # Prior revision still retrievable.
    assert client.get(f"/api/v1/datasets/{ds_id}/revisions/1").status_code == 200


def test_run_with_model_override_snapshots_key(client, png_bytes):
    ds_id = _dataset(client, png_bytes)
    r = client.post(f"/api/v1/datasets/{ds_id}/runs", json={"model_key": "finetuned"})
    assert r.status_code == 202
    rev = client.get(f"/api/v1/datasets/{ds_id}/revisions/{r.json()['number']}").json()
    assert rev["model_key"] == "finetuned"


def test_run_unknown_model_422(client, png_bytes):
    ds_id = _dataset(client, png_bytes)
    r = client.post(f"/api/v1/datasets/{ds_id}/runs", json={"model_key": "ghost"})
    assert r.status_code == 422
    assert "base" in r.json()["available_models"]


def test_run_missing_dataset_404(client):
    assert client.post("/api/v1/datasets/ds_nope/runs", json={}).status_code == 404


def test_missing_revision_404(client, png_bytes):
    ds_id = _dataset(client, png_bytes)
    assert client.get(f"/api/v1/datasets/{ds_id}/revisions/99").status_code == 404
