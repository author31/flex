"""Foundational contract — GET /models (named-model registry, feature 002)."""

from __future__ import annotations


def test_list_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    keys = {m["key"] for m in body["models"]}
    assert {"base", "finetuned"} <= keys
    assert body["default"] == "base"
    assert all("model_id" in m for m in body["models"])
