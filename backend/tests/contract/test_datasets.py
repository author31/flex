"""US1 contract — datasets (create / list / get)."""

from __future__ import annotations


def _upload(client, png_bytes) -> str:
    r = client.post("/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")})
    assert r.status_code == 201, r.text
    return r.json()["image_id"]


def test_create_and_get_dataset(client, png_bytes):
    img1, img2 = _upload(client, png_bytes), _upload(client, png_bytes)
    r = client.post(
        "/api/v1/datasets",
        json={
            "name": "smiles",
            "model_key": "base",
            "params": {"steps": 30},
            "items": [
                {"image_id": img1, "box": [10, 20, 40, 50], "prompt": "a wide smile"},
                {"image_id": img2, "region": "mouth", "prompt": "a grin"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    ds_id = r.json()["dataset_id"]

    got = client.get(f"/api/v1/datasets/{ds_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["model_key"] == "base"
    assert body["params"]["steps"] == 30
    assert [i["prompt"] for i in body["items"]] == ["a wide smile", "a grin"]
    assert body["items"][0]["box"] == [10, 20, 40, 50]
    assert body["items"][1]["region"] == "mouth"
    assert body["revisions"] == []


def test_list_datasets(client, png_bytes):
    img = _upload(client, png_bytes)
    client.post(
        "/api/v1/datasets",
        json={"model_key": "base", "items": [{"image_id": img, "box": [0, 0, 5, 5], "prompt": "x"}]},
    )
    r = client.get("/api/v1/datasets")
    assert r.status_code == 200
    rows = r.json()["datasets"]
    assert len(rows) >= 1
    assert {"dataset_id", "name", "model_key", "item_count", "latest_revision"} <= rows[0].keys()


def test_create_empty_dataset_422(client):
    r = client.post("/api/v1/datasets", json={"model_key": "base", "items": []})
    assert r.status_code == 422


def test_create_incomplete_row_422(client, png_bytes):
    img = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/datasets",
        json={"model_key": "base", "items": [{"image_id": img, "prompt": "no selection"}]},
    )
    assert r.status_code == 422


def test_create_unknown_model_422_lists_available(client, png_bytes):
    img = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/datasets",
        json={"model_key": "nope", "items": [{"image_id": img, "box": [0, 0, 5, 5], "prompt": "x"}]},
    )
    assert r.status_code == 422
    assert "base" in r.json()["available_models"]


def test_get_dataset_includes_image_dims(client, png_bytes):
    """Frontend rebuilds rows from saved items → needs image dimensions back."""
    img = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/datasets",
        json={"model_key": "base", "items": [{"image_id": img, "box": [0, 0, 5, 5], "prompt": "x"}]},
    )
    ds_id = r.json()["dataset_id"]
    item = client.get(f"/api/v1/datasets/{ds_id}").json()["items"][0]
    assert item["image_width"] == 128 and item["image_height"] == 128


def test_update_dataset_replaces_rows(client, png_bytes):
    img1, img2 = _upload(client, png_bytes), _upload(client, png_bytes)
    ds_id = client.post(
        "/api/v1/datasets",
        json={"name": "v1", "model_key": "base", "items": [
            {"image_id": img1, "box": [0, 0, 5, 5], "prompt": "first"},
        ]},
    ).json()["dataset_id"]

    # Update: new name, model, and a different row set.
    r = client.put(
        f"/api/v1/datasets/{ds_id}",
        json={"name": "v2", "model_key": "finetuned", "items": [
            {"image_id": img1, "box": [1, 1, 9, 9], "prompt": "edited"},
            {"image_id": img2, "region": "mouth", "prompt": "added"},
        ]},
    )
    assert r.status_code == 200, r.text

    got = client.get(f"/api/v1/datasets/{ds_id}").json()
    assert got["name"] == "v2" and got["model_key"] == "finetuned"
    assert [i["prompt"] for i in got["items"]] == ["edited", "added"]
    assert got["items"][0]["box"] == [1, 1, 9, 9]


def test_update_missing_dataset_404(client, png_bytes):
    img = _upload(client, png_bytes)
    r = client.put(
        "/api/v1/datasets/ds_nope",
        json={"model_key": "base", "items": [{"image_id": img, "box": [0, 0, 5, 5], "prompt": "x"}]},
    )
    assert r.status_code == 404


def test_update_preserves_revisions(client, png_bytes):
    img = _upload(client, png_bytes)
    ds_id = client.post(
        "/api/v1/datasets",
        json={"model_key": "base", "items": [{"image_id": img, "box": [0, 0, 5, 5], "prompt": "x"}]},
    ).json()["dataset_id"]
    client.post(f"/api/v1/datasets/{ds_id}/runs", json={})  # revision 1
    r = client.put(
        f"/api/v1/datasets/{ds_id}",
        json={"model_key": "base", "items": [{"image_id": img, "box": [0, 0, 6, 6], "prompt": "y"}]},
    )
    assert len(r.json()["revisions"]) == 1  # history kept across edits


def test_get_missing_dataset_404(client):
    assert client.get("/api/v1/datasets/ds_nope").status_code == 404
