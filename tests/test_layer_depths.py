"""Cross-model depth mapping must preserve layer-count and index conventions."""

import json
import pytest

from steer_opd.method_config import resolve_layer_depths, resolve_method, Method


@pytest.mark.parametrize("blocks,expected", [
    (28, (10, 13, 16, 20)),
    (32, (12, 15, 18, 23)),
    (40, (15, 19, 23, 29)),
])
def test_same_depths_map_to_each_students_actual_blocks(blocks, expected):
    assert resolve_layer_depths(blocks, [.4, .5, .6, .75]) == expected


def test_half_block_rounds_up_and_first_last_blocks_stay_in_bounds():
    assert resolve_layer_depths(25, [.5]) == (12,)
    assert resolve_layer_depths(25, [.58]) == (14,)
    assert resolve_layer_depths(28, [.001, 1.]) == (0, 27)


def test_small_model_collisions_cannot_silently_reduce_injection_layer_count():
    with pytest.raises(ValueError, match="collapse"):
        resolve_layer_depths(2, [.4, .5])
    with pytest.raises(ValueError, match="fractions"):
        resolve_layer_depths(28, [50])


def test_relative_config_resolves_to_existing_absolute_method_schema(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 28}))
    cfg = {"student": str(tmp_path), "steering_method": {"layer_depths": [.4, .5, .6, .75]}}
    assert resolve_method(cfg).to_dict() == Method(layers=(10, 13, 16, 20)).to_dict()
    cfg["steering_method"]["layers"] = [11, 13, 16, 20]
    with pytest.raises(ValueError, match="do not match"):
        resolve_method(cfg)
