"""Verify the external reference contract, source import and KL gradient direction."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from steer_opd.math_data import grade_rethinking_math, prompt_ids
from steer_opd.rethinking_protocol import (
    ROOT, build_overrides, compose_reference, execution_blockers,
    load_profile, reference_environment,
)


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def test_defaults_match_real_reference_shell_and_inherited_yaml():
    path = ROOT / "scripts/audit_rethinking_alignment.py"
    spec = importlib.util.spec_from_file_location("_alignment_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit()
    assert len(result["paper_default_checks"]) >= 70
    assert len(result["evaluation_checks"]) >= 6
    assert result["unexpected_differences"] == []
    assert not result["loss"]["newer_vendor_forward_kl_backend_used"]


def test_import_path_selects_the_fork_and_drops_the_old_hook():
    profile = load_profile()
    original = {"PYTHONPATH": str(ROOT / "vendor/verl"), "STEER_OPD_CONFIG": "old_loss_only.json"}
    env = reference_environment(profile, {**os.environ, **original})
    assert original["STEER_OPD_CONFIG"] == "old_loss_only.json"
    assert "STEER_OPD_CONFIG" not in env
    output = subprocess.check_output(
        [sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('verl').origin)"],
        cwd=ROOT, env=env, text=True,
    ).strip()
    assert Path(output).is_relative_to(ROOT / profile["reference"]["python_root"])
    values = build_overrides(profile, ROOT / "audit/rethinking_alignment_2026-11-12/test_preview")
    composed = compose_reference(profile, values)
    assert composed["algorithm"]["adv_estimator"] == "token_reward_direct"
    assert "distillation" not in composed


def test_pause_and_unsupported_controlled_policy_cannot_execute():
    profile = load_profile()
    assert any("paused" in reason for reason in execution_blockers(profile))
    # Even a resumed vanilla profile must not silently substitute loss-only steering.
    profile["execution_authorized"] = True
    profile["project"]["processed_inputs_prepared"] = True
    assert execution_blockers(profile, "E0") == []
    assert any("not implemented" in reason for reason in execution_blockers(profile, "E3"))
    with pytest.raises(ValueError, match="inside the project"):
        build_overrides(profile, "/tmp/outside_steer_opd")


def test_original_student_weighted_surrogate_gradient_is_not_teacher_forward_kl():
    reference = ROOT / "ref/ref_rep/Rethinking-OPD/verl/verl"
    namespace = {"torch": torch, "AlgoConfig": type("UnusedAlgoConfig", (), {})}
    source_functions(reference / "utils/torch_functional.py", ["masked_sum", "masked_mean"], namespace)
    namespace["verl_F"] = SimpleNamespace(masked_mean=namespace["masked_mean"])
    source_functions(reference / "trainer/ppo/core_algos.py",
                     ["agg_loss", "compute_policy_loss_vanilla"], namespace)
    source_functions(reference / "workers/actor/dp_actor.py", ["compute_reward_weights"], namespace)
    logits = torch.tensor([[[1.7, 0.4, -0.9], [0.1, 1.2, -0.4], [0.3, 0.0, 1.1]]],
                          dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.tensor([[[0.2, 1.4, -0.1], [1.1, -0.2, 0.4], [0.1, 1.4, -0.4]]],
                                  dtype=torch.float64, requires_grad=True)
    old_all = logits.detach().log_softmax(-1)
    ids = old_all.topk(2, dim=-1).indices
    old = old_all.gather(-1, ids)
    teacher_on_student = teacher_logits.detach().log_softmax(-1).gather(-1, ids)
    weights = namespace["compute_reward_weights"](
        old, teacher_on_student, torch.ones_like(old, dtype=torch.bool), "student_p")
    advantage = ((teacher_on_student - old) * weights).detach()
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    config = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                               "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    loss, _ = namespace["compute_policy_loss_vanilla"](
        old, logits.log_softmax(-1).gather(-1, ids), advantage, mask, config=config)
    loss.backward()
    # Analytical derivative at ratio=1, including full-softmax normalization.
    coefficients = -advantage
    expected = torch.zeros_like(logits).scatter(-1, ids, coefficients)
    expected -= old_all.exp() * coefficients.sum(-1, keepdim=True)
    expected *= mask.unsqueeze(-1) / (mask.sum() + 1e-8)
    torch.testing.assert_close(logits.grad, expected, rtol=1e-10, atol=1e-10)
    forward_kl_grad = (old_all.exp() - teacher_logits.detach().softmax(-1)) * mask.unsqueeze(-1) / mask.sum()
    assert not torch.allclose(logits.grad, forward_kl_grad)
    assert teacher_logits.grad is None
    assert torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))


def test_non_thinking_prompt_and_original_rule_grader():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/models/Qwen3-1.7B-Base", local_files_only=True)
    tokens = prompt_ids(tokenizer, "What is 1+1?", enable_thinking=False)
    assert tokenizer.decode(tokens).endswith("<think>\n\n</think>\n\n")
    assert grade_rethinking_math(r"336^\circ", r"Therefore $\boxed{336}$.")["correct"]
    assert not grade_rethinking_math("336", r"Therefore $\boxed{335}$.")["correct"]
