"""An experiment hold must reach and stop independently executable stages."""

import importlib
import sys

import pytest

from steer_opd.compass_campaign import validate_config
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.resumable_core import ROOT, read_json


@pytest.mark.parametrize("entry", ["panel_cli", "panel_execute", "generate", "score",
                                  "banks", "screen_cli", "screen_run"])
def test_selected_hold_blocks_each_stage_before_models_gpus_or_artifact_writes(entry, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    cfg = read_json(ROOT / "configs/compass_static_eight_20261113.json")
    assert cfg["execution_hold"] == "awaiting_user_approval"
    modules = {"panel_cli": "run_compass_static_panel", "panel_execute": "run_compass_static_panel",
               "generate": "figure8_generate", "score": "score_compass_sources",
               "banks": "prepare_compass_banks", "screen_cli": "run_static_overlap_screen",
               "screen_run": "run_static_overlap_screen"}
    module = importlib.import_module(modules[entry])
    monkeypatch.setattr(module, "read_json", lambda _: cfg)
    monkeypatch.setattr(module, "ROOT", tmp_path, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Held experiment reached model preparation or GPU subprocess")

    for name in ("load_codecs", "load_text_scorer", "require_gpu", "prepare", "prepare_plans"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    if entry == "panel_execute":
        action = lambda: module.execute(cfg)
    elif entry == "banks":
        action = lambda: module.build(cfg)
    elif entry == "screen_run":
        action = lambda: module.run(cfg, None, None, None, None)
    else:
        arguments = [modules[entry], "--config" if entry == "panel_cli" else "--job", "unused.json"]
        if entry != "generate":
            arguments.append("--execute")
        if entry == "score":
            arguments.extend(["--role", "teacher"])
        monkeypatch.setattr(sys, "argv", arguments)
        action = module.main
    with pytest.raises(RuntimeError, match="execution is held"):
        action()
    assert list(tmp_path.iterdir()) == []


def test_removing_user_hold_does_not_silently_ignore_protocol_blockers():
    cfg = read_json(ROOT / "configs/compass_static_eight_20261113.json")
    cfg.pop("execution_hold")
    with pytest.raises(RuntimeError, match="prerequisites remain"):
        require_execution_allowed(cfg)


@pytest.mark.parametrize("cap", ["max_num_seqs", "max_num_active_seqs", "vllm_max_num_seqs",
                               "cudagraph_capture_sizes", "max_cudagraph_capture_size"])
def test_new_panel_rejects_nested_manual_scheduler_caps(cap):
    cfg = read_json(ROOT / "configs/compass_static_eight_20261113.json")
    cfg["engine_options"] = {cap: 16}
    with pytest.raises(ValueError, match="scheduling at its defaults"):
        validate_config(cfg)


def test_standalone_score_rejects_scheduler_caps_before_input_preparation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("run_static_overlap_screen")
    monkeypatch.setattr(module, "read_json", lambda _: {"max_num_seqs": 16})
    monkeypatch.setattr(sys, "argv", ["screen", "--job", "unused.json", "--execute"])
    monkeypatch.setattr(module, "prepare", lambda _: pytest.fail("Manual cap reached preparation"))
    with pytest.raises(ValueError, match="scheduling at its defaults"):
        module.main()
