"""Semantic checks for causal matching and batched probe/official KV merging."""

import importlib.util

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.batched_controlled import BatchedController, batched_step, forward_rows
from steer_opd.controlled_rollout import next_distribution, generate_response
from steer_opd.gate_controls import matched_draw, offline_matched_mask, calibrate_threshold
from steer_opd.method_config import Method, intervention_phases
from steer_opd.resumable_core import ROOT
from steer_opd.resumable_core import response_hidden, score_hidden, response_loss
from steer_opd.resumable_core import atomic_json, atomic_torch, digest
from steer_opd.direction_bank import file_sha256
from steer_opd.direction_bank import make_bank, bank_vectors
from steer_opd.steering import CompositeSteeringController


def model(seed):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def test_random_matching_cannot_read_current_or_future_shadow_and_preserves_offline_bins():
    for shadow in [True, False]:
        history = {}
        decision, metric = matched_draw(history, shadow, 0, .3, .2, 4, True)
        assert decision is False and metric["matching_probability"] == .2
        assert history["0"] == [1, int(shadow)]
    original = [True, False, True, False, False, True, False, False]
    randomized = offline_matched_mask(original, 72, bin_width=4)
    assert [sum(randomized[i:i+4]) for i in [0, 4]] == [2, 1]
    result = calibrate_threshold([0, .25, .5, .75, 1], .2, "lt")
    assert result["observed_rate"] == .2


def test_asymmetric_phases_require_explicit_state_objective_and_cutoff_is_shared():
    with pytest.raises(ValueError, match="candidate_rkl"):
        intervention_phases({"steering_phase": "learning_only"}, "steer", 1)
    cfg = {"steering_phase": "learning_only", "loss_objective": "candidate_rkl", "assistance_until_step": 9}
    assert intervention_phases(cfg, "steer", 9) == (False, True)
    assert intervention_phases(cfg, "steer", 10) == (False, False)


@pytest.mark.parametrize("gate", ["all", "overlap", "matched_random", "matched_position_random"])
def test_batched_official_cache_matches_full_recorded_intervention_replay(gate):
    student, teacher = model(3), model(4)
    deltas = torch.randn(2, 16) * .3
    controller = BatchedController(student, {0: deltas})
    method = Method(layers=(0,), gate=gate, overlap_threshold=1, random_probability=.5)
    attention = torch.ones(2, 2, dtype=torch.bool)
    prefix = torch.tensor([[1, 2], [3, 4]])
    histories = [{}, {}]
    tokens = prefix[:, -1:]
    masks = []
    try:
        with torch.inference_mode():
            _, sc = forward_rows(student, prefix[:, :1], None, attention[:, :1])
            _, tc = forward_rows(teacher, prefix[:, :1], None, attention[:, :1])
            for step in range(4):
                p, q, sc, tc, inject, _, _ = batched_step(student, teacher, tokens, sc, tc, attention,
                    controller, method, [step, step], [.2, .8], histories, [True, True])
                masks.append(inject)
                replay_mask = torch.zeros_like(prefix, dtype=torch.bool)
                replay_mask[:, 1:] = torch.tensor(masks, dtype=torch.bool).T
                expected, _ = forward_rows(student, prefix, None, attention, controller, replay_mask)
                torch.testing.assert_close(p, expected, atol=3e-6, rtol=3e-6)
                next_ids = p.argmax(-1, keepdim=True)
                tokens = next_ids
                prefix = torch.cat([prefix, next_ids], dim=1)
                attention = torch.cat([attention, torch.ones(2, 1, dtype=torch.bool)], dim=1)
    finally:
        controller.detach()


def test_batched_generate_commits_native_topk_and_teacher_scoring(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_batched_generate", ROOT / "scripts/figure8_generate_batched.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Tokenizer:
        pad_token_id = 0
        def convert_tokens_to_ids(self, token):
            return None
        def decode(self, tokens, **kwargs):
            return str(tokens)
    requests = [{"key": "a", "id": "q1", "seed": 10, "prompt_token_ids": [1, 2]},
                {"key": "b", "id": "q2", "seed": 11, "prompt_token_ids": [3, 4, 5]}]
    job = {"method": Method(layers=(0,)).to_dict(), "max_tokens": 4, "support_size": 47,
           "save_rollout_top_k": 16, "teacher": "teacher", "model": "student"}
    captured = []
    monkeypatch.setattr(module, "commit_topk_response", lambda directory, row, score: captured.append((row, score)))
    student, teacher = model(1), model(2)
    module.run_batch(student, teacher, Tokenizer(), requests, job, "hash", tmp_path, None)
    assert len(captured) == 2
    assert all(len(row["token_ids"]) == 4 and not any(row["steering_mask"]) for row, _ in captured)
    assert all(score["teacher_logps"].shape == (4, 16) for _, score in captured)
    with torch.inference_mode():
        for row, score in captured:
            replay = score_hidden(student, response_hidden(student, row), candidates=score["ids"])
            expected = score_hidden(teacher, response_hidden(teacher, row), candidates=score["ids"])
            torch.testing.assert_close(replay["logps"], score["logps"], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(expected["logps"], score["teacher_logps"], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("objective", ["candidate_rkl", "compass_candidate", "compass_sampled", "eopd_top16_surrogate", "sampled_rkl", "eopd_sampled"])
def test_explicit_baseline_objectives_have_real_finite_student_gradients(objective):
    student, teacher, reference, anchor = model(5), model(6), model(7), model(8)
    row = {"prompt_token_ids": [1, 2], "token_ids": [3, 4, 5], "steering_mask": [False]*3}
    with torch.inference_mode():
        scores = score_hidden(student, response_hidden(student, row), sampled=row["token_ids"])
        target = score_hidden(teacher, response_hidden(teacher, row), candidates=scores["ids"], own_topk=True,
                              include_entropy=True, sampled=row["token_ids"])
        scores.update(teacher_logps=target["logps"], teacher_topk_ids=target["own_ids"],
                      teacher_topk_logps=target["own_logps"], teacher_entropy=target["entropy"], teacher_sampled_logps=target["sampled_logps"])
        for name, value in [("reference", reference), ("anchor", anchor)]:
            extra = score_hidden(value, response_hidden(value, row), candidates=scores["ids"], sampled=row["token_ids"])
            scores[f"{name}_logps"] = extra["logps"]
            scores[f"{name}_sampled_logps"] = extra["sampled_logps"]
    loss, _ = response_loss(student, row, scores, None, 3, objective=objective)
    assert torch.isfinite(torch.tensor(loss))
    assert sum(float(p.grad.float().square().sum()) for p in student.parameters() if p.grad is not None) > 0


def test_batched_partial_resume_with_different_request_lengths_preserves_rng_and_policies(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_batched_resume", ROOT / "scripts/figure8_generate_batched.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Tokenizer:
        pad_token_id = 0
        def convert_tokens_to_ids(self, token):
            return None
        def decode(self, tokens, **kwargs):
            return str(tokens)
    requests = [{"key": "a", "id": "q1", "seed": 20, "prompt_token_ids": [1, 2]},
                {"key": "b", "id": "q2", "seed": 21, "prompt_token_ids": [3, 4, 5]}]
    job = {"method": Method(layers=(0,)).to_dict(), "max_tokens": 34, "support_size": 47,
           "save_rollout_top_k": 16, "teacher": "teacher", "model": "student"}
    full, resumed = {}, {}
    monkeypatch.setattr(module, "commit_topk_response", lambda directory, row, score: full.update({row["key"]: (row, score)}))
    module.run_batch(model(1), model(2), Tokenizer(), requests, job, "hash", tmp_path / "full", None)
    original_save = module.atomic_torch
    def interrupted(path, value):
        original_save(path, value)
        if path.name == "a.pt" and path.parent.name == "progress":
            raise InterruptedError("Simulated kill after one request's32-token durable checkpoint")
    monkeypatch.setattr(module, "atomic_torch", interrupted)
    monkeypatch.setattr(module, "commit_topk_response", lambda directory, row, score: resumed.update({row["key"]: (row, score)}))
    with pytest.raises(InterruptedError):
        module.run_batch(model(1), model(2), Tokenizer(), requests, job, "hash", tmp_path / "resume", None)
    monkeypatch.setattr(module, "atomic_torch", original_save)
    module.run_batch(model(1), model(2), Tokenizer(), requests, job, "hash", tmp_path / "resume", None)
    for key in ["a", "b"]:
        assert full[key][0]["token_ids"] == resumed[key][0]["token_ids"]
        torch.testing.assert_close(full[key][1]["logps"], resumed[key][1]["logps"], atol=3e-6, rtol=3e-6)


def test_panel_declares_common_backend_and_keeps_first_pair_and_objectives_separate():
    from steer_opd.full_experiment import panel
    from steer_opd.method_config import resolve_method
    values = panel({}, {"layers": [6, 13], "rho": .03},
        {"overlap": {"threshold": .5, "observed_rate": .2}, "entropy": {"threshold": .1}, "late_start_token": 512})
    assert [(name, seed) for name, seed, _ in values[:2]] == [("vanilla", 42), ("main_222", 42)]
    assert len({cfg["run_dir"] for _, _, cfg in values}) == len(values)
    for _, _, cfg in values:
        resolve_method(cfg)
        intervention_phases(cfg, cfg["groups"][0], 1)
        assert cfg["generation_backend"] in ("hf_batched", "vllm")
        if cfg["generation_backend"] == "vllm":
            assert resolve_method(cfg).gate == "all"
        assert cfg["eval_steps"] in ([50], [])


def test_shared_initial_triplets_build_banks_without_model_and_reject_changed_text(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_shared_extract", ROOT / "scripts/figure8_extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_model", lambda *args: pytest.fail("Initial cached bank unnecessarily loaded a model"))
    method = Method(direction="T-R", layers=(0,))
    cfg = {"student": "base", "reference": "reference", "steering_method": method.to_dict()}
    question = {"id": "q1", "student_prompt_ids": [1, 2]}
    atomic_json(tmp_path / "config.json", cfg)
    atomic_json(tmp_path / "questions.json", [question])
    for role, text in [("teacher", "teacher answer"), ("reference", "reference answer")]:
        atomic_json(tmp_path / role / "q1.json", {"text": text})
    value = {"usable": True, "identity": {"question_hash": digest(question),
        "text_hashes": {"teacher": digest("teacher answer"), "reference": digest("reference answer")}},
        "roles": {"0": {role: {"pool": pool, "norm": torch.ones(2)} for role, pool in [
            ("teacher", torch.arange(16).float()), ("reference", torch.zeros(16))]}}}
    atomic_torch(tmp_path / "cache/q1.pt", value)
    atomic_json(tmp_path / "cache/manifest.json", {"checkpoint": "base", "step": 0,
        "file_sha256": {"q1": file_sha256(tmp_path / "cache/q1.pt")}})
    atomic_torch(tmp_path / "scales.pt", {"0": torch.tensor(1.)})
    job = {"config": "config.json", "questions": "questions.json", "output": "out", "method": method.to_dict(),
        "extractor_model": "base", "extractor_step": 0, "fixed_scales": "scales.pt", "shared_initial_pool": "cache",
        "role_directories": {"teacher": "teacher", "reference": "reference"}}
    module.extract_job(job)
    assert (tmp_path / "out/vectors.pt").exists()
    atomic_json(tmp_path / "teacher/q1.json", {"text": "changed answer"})
    with pytest.raises(ValueError, match="modified texts"):
        module.extract_job({**job, "output": "other"})


def test_per_question_random_vectors_do_not_collapse_to_one_global_vector_and_zero_is_not_amplified():
    pooled = {qid: {"usable": True, "layers": {"0": {"pools": [torch.arange(16).float()*scale, torch.zeros(16)],
        "norms": [torch.ones(2), torch.ones(2)]}}} for qid, scale in [("q1", 1.), ("q2", 2.), ("qzero", 0.)]}
    value = make_bank(pooled, Method(layers=(0,), granularity="per_question", direction_control="random"), {"step": 0})
    first, _ = bank_vectors(value, "q1")
    second, _ = bank_vectors(value, "q2")
    zero, _ = bank_vectors(value, "qzero")
    assert not torch.equal(first[0], second[0])
    assert float(first[0].norm()) == pytest.approx(float(second[0].norm()))
    assert torch.count_nonzero(zero[0]) == 0


def test_full_support_acceptance_checks_teacher_and_current_only_probe_on_controlled_history():
    spec = importlib.util.spec_from_file_location("_probe_accept", ROOT / "scripts/figure8_branch_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    student, teacher = model(20), model(21)
    controller = CompositeSteeringController(student, {0: torch.linspace(-.3, .3, 16)})
    try:
        row, score = generate_response(student, teacher, {"prompt_token_ids": [1, 2], "seed": 42}, controller,
            Method(layers=(0,), gate="overlap", overlap_threshold=1.), max_tokens=4, support_size=47, k=16,
            stop_ids=set(), audit_full=True)
        assert module.verify_full_replay(student, row, score, controller)["passed"]
        assert module.verify_full_replay(teacher, row, {"full_rollout_logps": score["full_teacher_logps"]}, None)["passed"]
        for position in range(4):
            probe_row = {**row, "token_ids": row["token_ids"][:position+1],
                         "steering_mask": row["steering_mask"][:position]+[False]}
            assert module.verify_full_replay(student, probe_row,
                {"full_rollout_logps": score["full_probe_logps"][position:position+1]}, controller, last_only=True)["passed"]
    finally:
        controller.detach()
