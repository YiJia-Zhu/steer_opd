"""Review regressions: masks and cached banks must match the named control."""

import importlib.util

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.direction_bank import file_sha256, make_bank
from steer_opd.method_config import Method, resolve_method
from steer_opd.overlap_tuning import evaluate_candidate
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, response_hidden, score_hidden


def script(name):
    spec = importlib.util.spec_from_file_location(f"_coverage_{name}", ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def bank(layers=(0,), checkpoint="model", step=0):
    method = Method(layers=layers)
    pooled = {"q1": {"usable": True, "layers": {str(layer): {
        "pools": [torch.arange(16).float() + 1 + layer, torch.zeros(16)],
        "norms": [torch.tensor([1.0]), torch.tensor([1.0])]}
        for layer in layers}}}
    return make_bank(pooled, method, {"checkpoint": checkpoint, "step": step})


@pytest.mark.parametrize("phase", ["rollout_only", "loss_only", "last_10_percent_off"])
def test_unimplemented_injection_variants_are_rejected_instead_of_mislabeled(phase):
    with pytest.raises(ValueError, match="phase/schedule is not implemented"):
        resolve_method({"injection": phase})


@pytest.mark.parametrize("changed", ["method", "checkpoint", "step"])
def test_cached_bank_reuse_checks_scientific_identity_even_when_file_hash_matches(tmp_path, monkeypatch, changed):
    module = script("run_figure8_campaign")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run = tmp_path / "run"
    campaign.cfg = {"student": "model", "steering_method": {"layers": [0]}}
    target = campaign.run / "directions/initial/vectors.pt"
    metadata_path = target.parent / "metadata.json"
    value = bank()
    def save():
        atomic_torch(target, value)
        atomic_json(metadata_path, {"method": value["method"], "vector_file_sha256": file_sha256(target)})
    save()
    assert campaign.prepare_directions(1) == target
    if changed == "method":
        value["method"]["granularity"] = "per_question"
    elif changed == "checkpoint":
        value["identity"]["checkpoint"] = "another_model"
    else:
        value["identity"]["step"] = 1
    save()
    with pytest.raises(ValueError, match="method differs|checkpoint/step mismatch"):
        campaign.prepare_directions(1)


def test_mixed_mask_two_layer_diagnostic_logs_actual_mask_and_repairs_old_metadata(tmp_path, monkeypatch):
    torch.manual_seed(57)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    student, teacher = Qwen3ForCausalLM(cfg).eval(), Qwen3ForCausalLM(cfg).eval()
    rows = [{"id": "q1", "prompt_token_ids": [1, 2], "token_ids": [3, 4, 5],
             "steering_mask": [True, False, True]}]
    teacher_dir = tmp_path / "teacher"
    with torch.inference_mode():
        atomic_torch(teacher_dir / "q1.pt", score_hidden(teacher, response_hidden(teacher, rows[0]), include_entropy=True))
    value = bank(layers=(0, 1))
    baseline = evaluate_candidate(student, value, {"id": "base", "kind": "baseline"}, rows,
                                  teacher_dir, tmp_path / "base")
    assert not baseline["q1"]["steering_mask"].any()
    condition = {"id": "mixed", "kind": "bank", "layers": [0, 1], "rho": 0.03}
    observed = evaluate_candidate(student, value, condition, rows, teacher_dir, tmp_path / "mixed", tmp_path / "base")
    record = observed["q1"]
    assert record["steering_mask"].tolist() == [True, False, True]
    original_logps = record["logps"].clone()
    record["steering_mask"] = torch.ones(3, dtype=torch.bool)
    atomic_torch(tmp_path / "mixed/q1.pt", record)
    monkeypatch.setattr(student.model, "forward", lambda *a, **k: pytest.fail("Metadata repair repeated a model pass"))
    repaired = evaluate_candidate(student, value, condition, rows, teacher_dir, tmp_path / "mixed", tmp_path / "base")["q1"]
    assert repaired["steering_mask"].tolist() == [True, False, True]
    assert repaired["mask_recording_corrected"] and torch.equal(repaired["logps"], original_logps)
