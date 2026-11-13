"""Token evidence can be reused without silently losing positions or entropy."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.controlled_rollout import generate_response
from steer_opd.direction_bank import make_bank
from steer_opd.extraction import extract_pairs, load_token_activation_cache
from steer_opd.method_config import Method
from steer_opd.overlap_tuning import evaluate_candidate
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, project_logits, read_json, response_hidden, score_hidden
from steer_opd.rollout_topk import SOURCE, attach_token_entropy, commit_topk_response, load_rollout_topk
from steer_opd.steering import CompositeSteeringController
from steer_opd.token_statistics import prepare_teacher_statistics, save_trajectory_statistics


def tiny(seed=25):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


class Tokenizer:
    all_special_ids = []

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 47 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def roles(tmp_path):
    directories = {role: tmp_path / role for role in ["teacher", "student"]}
    for role, text in [("teacher", "abc<think>de</think>f"), ("student", "wxyz")]:
        atomic_json(directories[role] / "q1.json", {"id": "q1", "text": text, "finish_reason": "stop"})
    return directories


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(f"_recording_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_activation_cache_preserves_body_pooling_and_first_prediction_boundary(tmp_path):
    model = tiny()
    questions = [{"id": "q1", "student_prompt_ids": [1, 2, 3]}]
    method = Method(layers=(0, 1))
    output = tmp_path / "directions"
    pooled = extract_pairs(model, Tokenizer(), questions, roles(tmp_path), method, output,
                           {"checkpoint": "S0", "step": 0}, cache_token_activations=True, cache_dtype="float32")
    value = pooled["q1"]
    assert value["usable"] and set(value["token_activations"]) == {"teacher", "student"}
    for index, role in enumerate(method.roles):
        cache = load_token_activation_cache(output, value["token_activations"][role])
        n = len(cache["student_response_ids"])
        assert cache["state_input_positions"][0] == 2
        assert cache["prediction_state_rows"].tolist() == list(range(n))
        assert cache["response_token_state_rows"].tolist() == list(range(1, n + 1))
        assert cache["response_offsets"].shape == (n, 2)
        for layer in method.layers:
            states = cache["states"][str(layer)]
            assert states.shape == (n + 1, 16)
            body = states[cache["response_token_state_rows"]][cache["body_mask"]]
            torch.testing.assert_close(body.mean(0), value["layers"][str(layer)]["pools"][index], rtol=0, atol=0)
        if role == "teacher":
            assert not cache["body_mask"].all()  # Structural markup is mapped, not silently discarded.
    bank = make_bank(pooled, method, {"checkpoint": "S0", "step": 0})
    assert "q1" in bank["per_question"]


def test_cache_corruption_is_rejected_without_replaying_model(tmp_path, monkeypatch):
    model = tiny()
    question = [{"id": "q1", "student_prompt_ids": [1, 2]}]
    directories = roles(tmp_path)
    output = tmp_path / "directions"
    identity = {"checkpoint": "S0", "step": 0}
    pooled = extract_pairs(model, Tokenizer(), question, directories, Method(layers=(0,)), output, identity,
                           cache_token_activations=True)
    artifact = pooled["q1"]["token_activations"]["teacher"]
    with (output / artifact["path"]).open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(model.model, "forward", lambda *a, **k: pytest.fail("Replayed before validating cache"))
    with pytest.raises(ValueError, match="missing or changed"):
        extract_pairs(model, Tokenizer(), question, directories, Method(layers=(0,)), output, identity,
                      cache_token_activations=True)
    with pytest.raises(ValueError, match="missing or changed"):
        load_token_activation_cache(output, artifact)


def test_completed_extraction_metadata_can_be_repaired_without_loading_a_model(tmp_path, monkeypatch):
    module = load_script("figure8_extract", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    cfg = {"student": "S0", "shared_vocabulary_size": 47, "layer": 0, "rho": 0.03,
           "steering_method": {"direction": "T-S", "layers": [0]}}
    atomic_json(tmp_path / "config.json", cfg)
    atomic_json(tmp_path / "questions.json", [{"id": "q1", "student_prompt_ids": [1, 2]}])
    roles(tmp_path)
    job = {"config": "config.json", "questions": "questions.json", "output": "directions",
           "role_directories": {"teacher": "teacher", "student": "student"},
           "extractor_model": "S0", "extractor_step": 0, "fixed_scales": "scales.pt",
           "minimum_pairs": 1, "cache_token_activations": True, "token_activation_dtype": "float32"}
    monkeypatch.setattr(module, "load_model", lambda *a: tiny())
    monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
    module.extract_job(job)
    original = read_json(tmp_path / "directions/metadata.json")
    (tmp_path / "directions/metadata.json").unlink()
    monkeypatch.setattr(module, "load_model", lambda *a: pytest.fail("Metadata repair loaded a model"))
    module.extract_job(job)
    assert read_json(tmp_path / "directions/metadata.json") == original


def test_entropy_is_computed_on_full_frozen_support_not_conditioned_top16():
    model = tiny()
    model._opd_vocab_size = 31
    row = {"prompt_token_ids": [1, 2], "token_ids": [3, 4, 5]}
    with torch.inference_mode():
        hidden = response_hidden(model, row)
        score = score_hidden(model, hidden, include_entropy=True)
        lp = project_logits(model, hidden).log_softmax(-1)
        expected = -(lp.exp() * lp).sum(-1)
        conditional = score["logps"].log_softmax(-1)
        conditional_entropy = -(conditional.exp() * conditional).sum(-1)
    torch.testing.assert_close(score["entropy"], expected)
    assert score["entropy_support_size"] == 31
    assert (score["entropy"] > conditional_entropy + 0.1).all()


def test_teacher_entropy_is_filled_once_then_reused_with_original_top16(tmp_path):
    model = tiny()
    model._opd_vocab_size = 47
    rows = [{"id": "q1", "prompt_token_ids": [1, 2], "token_ids": [3, 4, 5]}]
    source, output = tmp_path / "source", tmp_path / "statistics"
    with torch.inference_mode():
        score = score_hidden(model, response_hidden(model, rows[0]))
    atomic_torch(source / "q1.pt", {"ids": score["ids"], "logps": score["logps"]})
    calls = []
    def loader():
        calls.append(True)
        return model
    prepare_teacher_statistics(rows, source, output, teacher_identity="T-revision", support_size=47, loader=loader)
    assert len(calls) == 1
    saved = torch.load(output / "q1.pt", weights_only=True)
    assert torch.equal(saved["ids"], score["ids"]) and torch.equal(saved["logps"], score["logps"])
    assert saved["entropy"].shape == (3,)
    prepare_teacher_statistics(rows, source, output, teacher_identity="T-revision", support_size=47,
                               loader=lambda: pytest.fail("Repeated teacher scoring"))
    assert torch.equal(torch.load(output / "q1.pt", weights_only=True)["entropy"], saved["entropy"])


def vllm_score():
    torch.manual_seed(31)
    lp = torch.randn(2, 47).log_softmax(-1)
    values, ids = lp.topk(16)
    return {"ids": ids.int(), "logps": values, "sampled_logps": lp[:, 1],
            "source": SOURCE, "support_size": 47, "k": 16, "probability_mode": "raw_logprobs"}, lp


@pytest.mark.parametrize("change", [{"positions": [0, 2]}, {"support_size": 46},
                                   {"source": "conditional_top16"}, {"entropy": [float("nan"), 1.0]}])
def test_worker_entropy_contract_rejects_wrong_positions_support_or_source(change):
    score, lp = vllm_score()
    statistics = {"positions": [0, 1], "support_size": 47, "source": "raw_full_support_logits",
                  "entropy": (-(lp.exp() * lp).sum(-1)).tolist(), **change}
    with pytest.raises(ValueError):
        attach_token_entropy(score, [1, 1], statistics)


def test_token_entropy_and_mask_are_bound_to_committed_rollout(tmp_path):
    score, lp = vllm_score()
    attach_token_entropy(score, [1, 1], {"positions": [0, 1], "support_size": 47,
        "source": "raw_full_support_logits", "entropy": (-(lp.exp() * lp).sum(-1)).tolist()})
    row = commit_topk_response(tmp_path, {"key": "a", "token_ids": [1, 1], "steering_mask": [True, True],
                                        "sampled_logps": score["sampled_logps"].tolist()}, score)
    loaded = load_rollout_topk(tmp_path, row, 16, 47)
    loaded["student_entropy"][0] += 0.01
    atomic_torch(tmp_path / row["rollout_topk"], loaded)
    with pytest.raises(ValueError, match="integrity/provenance"):
        load_rollout_topk(tmp_path, row, 16, 47)


def test_production_token_statistics_survive_score_cleanup_and_keep_probe_distinct(tmp_path):
    student, teacher = tiny(41), tiny(42)
    controller = CompositeSteeringController(student, {0: torch.linspace(-0.2, 0.2, 16)})
    request = {"id": "q1", "key": "a", "prompt_token_ids": [1, 2], "seed": 42,
               "vector_id": "frozen-vector", "vector_hash": "frozen-sha"}
    row, score = generate_response(student, teacher, request, controller,
        Method(layers=(0,), gate="overlap", overlap_threshold=1.0), max_tokens=4, support_size=47, k=16, stop_ids=set())
    controller.detach()
    row = commit_topk_response(tmp_path / "rollouts", row, score)
    target = tmp_path / "token_statistics/a.pt"
    save_trajectory_statistics(target, row, score, "T-revision", 47)
    atomic_torch(tmp_path / "temporary_score.pt", score)
    (tmp_path / "temporary_score.pt").unlink()
    saved = torch.load(target, weights_only=True)
    assert saved["prediction_input_positions"].tolist() == [1, 2, 3, 4]
    assert saved["vector_id"] == "frozen-vector" and saved["current_step_probe"]
    assert saved["entropy_available"] == {"student_entropy": True, "teacher_entropy": True, "probe_entropy": True}
    assert torch.equal(saved["steering_mask"], torch.tensor(row["steering_mask"]))
    torch.testing.assert_close(saved["teacher_on_student_logps"], score["teacher_logps"])
    expected = (score["probe_topk_ids"][:, :, None] == score["teacher_topk_ids"][:, None, :]).any(-1).float().mean(-1)
    torch.testing.assert_close(saved["probe_overlap16"], expected)
    assert (tmp_path / "rollouts" / row["rollout_topk"]).exists()


def test_first_grid_has_no_old_new_dataset_comparison(monkeypatch):
    module = load_script("run_pre_opd_overlap_tuning", monkeypatch)
    conditions = module.grid_conditions(read_json(ROOT / "configs/pre_opd_overlap_tuning.json"))
    assert len(conditions) == 16  # 15 layer/rho settings + alpha0.
    assert all(c["kind"] == "bank" for c in conditions)
