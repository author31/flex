"""Polish — full flow integration: upload → segment → edit(preset) → comparison.

Covers SC-005 across all 4 expressions × detail regions and SC-002 identity.
"""

from __future__ import annotations

import itertools

import pytest

EXPRESSIONS = ["smile", "angry", "sad", "surprised"]
REGIONS = ["eyes", "mouth", "eyebrows"]


def _upload(client, png_bytes) -> str:
    return client.post(
        "/api/v1/images", files={"file": ("c.png", png_bytes, "image/png")}
    ).json()["image_id"]


@pytest.mark.parametrize("preset,region", itertools.product(EXPRESSIONS, REGIONS))
def test_full_flow_all_combinations(client, png_bytes, preset, region):
    image_id = _upload(client, png_bytes)
    client.post(f"/api/v1/images/{image_id}/segments")
    edit_id = client.post(
        "/api/v1/edits", json={"image_id": image_id, "region": region, "preset": preset}
    ).json()["edit_id"]

    comp = client.get(f"/api/v1/edits/{edit_id}/comparison").json()
    assert comp["status"] == "completed"
    assert comp["region"] == region
    assert comp["metrics"]["identity_preserved"] is True  # SC-002
