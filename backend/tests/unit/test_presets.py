"""US3 domain test — preset_to_prompt mapping."""

from __future__ import annotations

import pytest

from app.domain import ExpressionPreset, FacialRegion, preset_to_prompt


@pytest.mark.parametrize("preset", list(ExpressionPreset))
def test_preset_maps_to_nonempty_prompt(preset):
    prompt = preset_to_prompt(preset, FacialRegion.MOUTH)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_preset_includes_region_focus():
    assert "eyes" in preset_to_prompt(ExpressionPreset.SURPRISED, FacialRegion.EYES)
    assert "mouth" in preset_to_prompt(ExpressionPreset.SMILE, FacialRegion.MOUTH)


def test_distinct_presets_distinct_prompts():
    prompts = {preset_to_prompt(p, FacialRegion.FACE) for p in ExpressionPreset}
    assert len(prompts) == len(list(ExpressionPreset))
