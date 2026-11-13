"""Real tiny-model causal probes, versioned directions and zero-update boundary."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.controlled_rollout import controlled_step, fork_cache, generate_response, next_distribution
from steer_opd.direction_bank import bank_vectors, make_bank
from steer_opd.method_config import Method, resolve_method, uses_legacy_direction
from steer_opd.overlap_tuning import evaluate_candidate, paired_summary
from steer_opd.resumable_core import (ROOT, atomic_json, atomic_torch, digest, original_advantage,
                                      original_token_losses, project_logits, read_json, response_hidden, response_loss, score_hidden)
from steer_opd.rollout_topk import commit_topk_response, load_rollout_topk
from steer_opd.steering import CompositeSteeringController, response_prediction_mask
from steer_opd.extraction import extract_pairs


def tiny(seed=1):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(f"_branches_{name}", ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def bank(method=None):
    method = method or Method(layers=(0, 1))
    pooled = {qid: {"usable": True, "layers": {
        str(layer): {"pools": [torch.arange(16).float() + shift, torch.arange(16).float() * 0.5],
                     "norms": [torch.tensor([10.0, 12.0]), torch.tensor([11.0])]}
        for layer in method.layers}} for qid, shift in [("q1", 1), ("q2", -2)]}
    return make_bank(pooled, method, {"checkpoint": "base", "step": 0})


def test_first_experiment_defaults_and_frozen_original_config_are_unchanged():
    cfg = read_json(ROOT / "configs/figure8_shared_top16_v2.json")
    method = resolve_method(cfg)
    assert (method.direction, method.gate, method.granularity, method.extractor) == ("T-S", "all", "global", "initial")
    assert uses_legacy_direction(cfg)
    assert digest(cfg) == read_json(ROOT / "data/figure8_shared_top16_v2_20261112/manifest.json")["config_hash"]


@pytest.mark.parametrize("options", [{"direction": "bad"}, {"gate": "overlap"}, {"gate": "entropy"},
                                    {"gate": "random"}, {"gate": "late"}, {"rho": float("nan")},
                                    {"layers": (1, 1)}, {"vector_assignment": "other_question"}])
def test_unfrozen_or_invalid_scientific_branches_are_rejected(options):
    with pytest.raises(ValueError):
        Method(**options)


def test_direction_roles_and_independent_refresh_axes():
    for direction, roles in [("T-S", ("teacher", "student")), ("T-R", ("teacher", "reference")),
                             ("R-S", ("reference", "student"))]:
        assert Method(direction=direction).roles == roles
    method = Method(granularity="per_question", extractor="current", refresh_interval=3)
    assert [method.extraction_step(step) for step in range(1, 8)] == [0, 0, 0, 3, 3, 3, 6]
    with pytest.raises(ValueError, match="explicit frozen reference"):
        resolve_method({"steering_method": {"direction": "T-R"}})


def test_raw_mean_before_normalization_per_question_and_equal_total_strength():
    value = bank()
    expected = torch.stack([r["0"]["direction"] for r in value["per_question"].values()]).mean(0)
    torch.testing.assert_close(value["layers"]["0"]["direction"], expected)
    vectors, vector_id = bank_vectors(value)
    assert vector_id and set(vectors) == {0, 1}
    total = sum(float(vectors[layer].norm() / value["layers"][str(layer)]["scale"]) ** 2 for layer in vectors)
    assert total == pytest.approx(value["method"]["rho"] ** 2)
    cached = bank(Method(layers=(0,), granularity="per_question"))
    a, _ = bank_vectors(cached, "q1")
    b, _ = bank_vectors(cached, "q2")
    assert not torch.equal(a[0], b[0])
    with pytest.raises(ValueError, match="No usable vector"):
        bank_vectors(cached, "unknown")


def test_other_question_and_negative_random_controls_preserve_norm_protocol():
    own = bank(Method(layers=(0,), granularity="per_question"))
    other = copy.deepcopy(own)
    other["method"]["vector_assignment"] = "other_question"
    a, _ = bank_vectors(own, "q2")
    b, _ = bank_vectors(other, "q1")
    torch.testing.assert_close(a[0], b[0])
    positive, _ = bank_vectors(own, "q1")
    negative, _ = bank_vectors(own, "q1", control="negative")
    random, _ = bank_vectors(own, "q1", control="random", seed=8)
    torch.testing.assert_close(negative[0], -positive[0])
    torch.testing.assert_close(random[0].norm(), positive[0].norm())


def test_probe_fork_does_not_modify_controlled_history_and_matches_explicit_replay():
    student, teacher = tiny(1), tiny(2)
    controller = CompositeSteeringController(student, {0: torch.linspace(-0.8, 0.8, 16)})
    with torch.inference_mode():
        _, history = next_distribution(student, [1, 2, 3], None, controller, True)
        original = [layer.keys.clone() for layer in history.layers]
        method = Method(layers=(0,), gate="overlap", overlap_threshold=1.0)
        p, _, official, _, inject, metrics, probe = controlled_step(
            student, teacher, [4], history, None, controller, method, 1, 0.5)
        assert inject and metrics["probe_overlap16"] < 1
        assert official.get_seq_length() == 4
        # The probe container is distinct; the official append changes history,
        # but its historical prefix stays byte-identical to the controlled KV.
        for old, layer in zip(original, official.layers):
            assert torch.equal(old, layer.keys[..., :3, :])
        full_row = {"prompt_token_ids": [1, 2, 3], "token_ids": [4, 5], "steering_mask": [True, True]}
        replay = project_logits(student, response_hidden(student, full_row, controller)).log_softmax(-1)
        torch.testing.assert_close(p, replay[-1], rtol=1e-5, atol=1e-5)
        assert not torch.equal(p, probe)
        forked = fork_cache(official)
        next_distribution(student, [6], forked)
        assert official.get_seq_length() == 4 and forked.get_seq_length() == 5
    controller.detach()


@pytest.mark.parametrize("gate,extra", [("all", {}), ("overlap", {"overlap_threshold": 0.0}),
                                        ("overlap", {"overlap_threshold": 1.0}),
                                        ("entropy", {"entropy_threshold": 0.0}),
                                        ("random", {"random_probability": 0.5}), ("late", {"start_token": 3})])
def test_actual_cached_rollout_probabilities_match_full_replay_with_recorded_mask(gate, extra):
    student, teacher = tiny(3), tiny(4)
    controller = CompositeSteeringController(student, {1: torch.linspace(-0.2, 0.2, 16)})
    method = Method(layers=(1,), gate=gate, **extra)
    row, scores = generate_response(student, teacher, {"id": "q1", "key": "a", "prompt_token_ids": [1, 2, 3], "seed": 17},
         controller, method, max_tokens=7, support_size=47, k=16, stop_ids=set())
    with torch.inference_mode():
        full = project_logits(student, response_hidden(student, row, controller)).log_softmax(-1)
        teacher_full = project_logits(teacher, response_hidden(teacher, row)).log_softmax(-1)
    torch.testing.assert_close(full.gather(-1, scores["ids"].long()), scores["logps"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(teacher_full.gather(-1, scores["ids"].long()), scores["teacher_logps"], rtol=1e-5, atol=1e-5)
    if gate == "late":
        assert row["steering_mask"] == [False] * 3 + [True] * 4
    if gate == "overlap" and extra["overlap_threshold"] == 0:
        assert not any(row["steering_mask"])
    controller.detach()


def test_interrupted_gated_rollout_restores_mask_rng_and_controlled_kv():
    student, teacher = tiny(5), tiny(6)
    controller = CompositeSteeringController(student, {0: torch.linspace(-0.3, 0.3, 16)})
    request = {"id": "q1", "key": "a", "prompt_token_ids": [1, 2, 3], "seed": 18}
    method = Method(layers=(0,), gate="random", random_probability=0.5)
    complete, expected = generate_response(student, teacher, request, controller, method,
                       max_tokens=8, support_size=47, k=16, stop_ids=set())
    saved = []
    def interrupted(state):
        saved.append(copy.deepcopy(state))
        raise OSError("simulated kill")
    with pytest.raises(OSError):
        generate_response(student, teacher, request, controller, method, max_tokens=8, support_size=47,
                          k=16, stop_ids=set(), save_progress=interrupted, save_every=3)
    resumed, actual = generate_response(student, teacher, request, controller, method,
                       max_tokens=8, support_size=47, k=16, stop_ids=set(), progress=saved[0])
    assert resumed["token_ids"] == complete["token_ids"] and resumed["steering_mask"] == complete["steering_mask"]
    torch.testing.assert_close(actual["logps"], expected["logps"], rtol=1e-5, atol=1e-5)
    controller.detach()


def test_recorded_gate_mask_is_shifted_and_gradient_replay_keeps_all_loss_positions():
    assert response_prediction_mask(3, 4, [True, False, True, False]).tolist() == [[False, False, True, False, True, False]]
    student, teacher = tiny(7), tiny(8)
    other = copy.deepcopy(student)
    delta = torch.linspace(-0.1, 0.1, 16)
    controller = CompositeSteeringController(student, {0: delta})
    row, scores = generate_response(student, teacher, {"prompt_token_ids": [1, 2], "seed": 19}, controller,
              Method(layers=(0,), gate="late", start_token=2), max_tokens=5, support_size=47, k=16, stop_ids=set())
    controller.detach()
    student.train().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    a = CompositeSteeringController(student, {0: delta})
    loss, _ = response_loss(student, row, scores, a, 5, 2)
    a.detach()
    b = CompositeSteeringController(other, {0: delta})
    inputs = torch.tensor([row["prompt_token_ids"] + row["token_ids"][:-1]])
    with b.loss_scope(response_prediction_mask(2, 5, row["steering_mask"])):
        current = project_logits(other, other.model(inputs, use_cache=False).last_hidden_state[0, 1:]).log_softmax(-1).gather(-1, scores["ids"].long())
        expected = original_token_losses(current, current.detach(), original_advantage(scores["logps"], scores["teacher_logps"])).mean()
        expected.backward()
    b.detach()
    assert loss == pytest.approx(float(expected.detach()))
    for x, y in zip(student.parameters(), other.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-7)


def test_online_teacher_scores_are_reused_without_a_second_teacher_forward(tmp_path, monkeypatch):
    stage = module("figure8_stage", monkeypatch)
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    student, teacher = tiny(9), tiny(10)
    hook = CompositeSteeringController(student, {0: torch.linspace(-0.1, 0.1, 16)})
    request = {"id": "q1", "key": "a", "prompt_token_ids": [1, 2], "seed": 20}
    row, score = generate_response(student, teacher, request, hook, Method(layers=(0,)),
                                  max_tokens=3, support_size=47, k=16, stop_ids=set())
    hook.detach()
    score.update(teacher_model="teacher", teacher_support_size=47)
    out = tmp_path / "steer/steps/0001/rollouts"
    atomic_json(out / "job.json", {"job": {"requests": [request]}})
    row = commit_topk_response(out, row, score)
    assert load_rollout_topk(out, row, 16, 47)["source"].startswith("hf_")
    monkeypatch.setattr(stage, "load_model", lambda *a, **k: pytest.fail("Teacher forward was repeated"))
    stage.teacher_scores({"run_dir": str(tmp_path), "teacher": "teacher", "supervision_top_k": 16,
                         "shared_vocabulary_size": 47, "student_topk_source": "rollout", "input_protocol": "exact_student_tokens"}, "steer", 1)
    assert read_json(tmp_path / "steer/steps/0001/rollout_overlap.json")["positions"] == 3


def test_fixed_prefix_diagnostic_cannot_call_backward_or_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("Diagnostic called backward"))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Diagnostic created optimizer"))
    student, teacher = tiny(11), tiny(12)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    rows = [{"id": "q1", "prompt_token_ids": [1, 2], "token_ids": [3, 4, 5]}]
    teacher_dir = tmp_path / "teacher"
    with torch.inference_mode():
        atomic_torch(teacher_dir / "q1.pt", score_hidden(teacher, response_hidden(teacher, rows[0])))
    value = bank()
    baseline = evaluate_candidate(student, value, {"id": "base", "kind": "baseline"}, rows, teacher_dir, tmp_path / "base")
    zero = evaluate_candidate(student, value, {"id": "zero", "kind": "bank", "layers": [0], "rho": 0.0},
                              rows, teacher_dir, tmp_path / "zero", tmp_path / "base")
    assert zero["q1"]["hook_removal_exact"]
    report = paired_summary(zero, baseline, bootstrap_samples=20)
    assert report["question_weighted_delta_pp"] == 0 and report["parameter_updates"] == 0
    assert all(torch.equal(value, before[key]) for key, value in student.state_dict().items())


def test_independent_diagnostic_prepares_disjoint_select_confirm_and_blocks_training(tmp_path, monkeypatch):
    tuning = module("run_pre_opd_overlap_tuning", monkeypatch)
    cfg = read_json(ROOT / "configs/pre_opd_overlap_tuning.json")
    cfg["run_dir"] = str(tmp_path)
    tuning.prepare(cfg, "configs/pre_opd_overlap_tuning.json")
    frozen = read_json(tmp_path / "diagnostic_manifest.json")
    assert len(frozen["select_ids"]) == len(frozen["confirm_ids"]) == 16
    assert not set(frozen["select_ids"]) & set(frozen["confirm_ids"])
    assert read_json(tmp_path / "campaign.json")["parameter_updates"] == 0
    campaign = module("run_figure8_campaign", monkeypatch)
    atomic_json(tmp_path / "request.json", cfg)
    with pytest.raises(ValueError, match="zero-update diagnostic"):
        campaign.Campaign(tmp_path / "request.json", resume_paused=True)


def test_variant_creation_requires_frozen_threshold_and_never_mutates_first_config(monkeypatch):
    creator = module("create_steering_variant", monkeypatch)
    base = read_json(ROOT / "configs/figure8_shared_top16_v2.json")
    original = copy.deepcopy(base)
    preset = read_json(ROOT / "configs/steering_branch_presets.json")["presets"]["main_222"]
    with pytest.raises(ValueError):
        creator.build_variant(base, preset, "test_222", reference="reference")
    cfg = creator.build_variant(base, preset, "test_222", reference="reference", overlap_threshold=0.4)
    assert not cfg["authorized"] and cfg["execution_hold"]
    assert cfg["data_dir"] != base["data_dir"] and base == original
    assert resolve_method(cfg).granularity == "per_question"


@pytest.mark.parametrize("gate", ["all", "overlap"])
def test_campaign_dispatch_binds_each_question_vector_and_selects_real_backend(tmp_path, monkeypatch, gate):
    campaign_module = module("run_figure8_campaign", monkeypatch)
    monkeypatch.setattr(campaign_module, "ROOT", tmp_path)
    cfg = {**read_json(ROOT / "configs/figure8_shared_top16_v2.json"),
           "student": "model", "teacher": "teacher", "reference": "reference", "shared_vocabulary_size": 47,
           "prompt_batch_size": 2, "rollouts_per_prompt": 1,
           "steering_method": {"direction": "T-R", "layers": [0], "granularity": "per_question", "extractor": "current",
                               "gate": gate, **({"overlap_threshold": 0.4} if gate == "overlap" else {})}}
    value = bank(resolve_method(cfg))
    vector = tmp_path / "directions/vectors.pt"
    atomic_torch(vector, value)
    rows = [{"id": key, "answer": "1", "student_prompt_ids": [1, 2]} for key in ["q1", "q2"]]
    atomic_json(tmp_path / "data/opd.json", rows)
    campaign = campaign_module.Campaign.__new__(campaign_module.Campaign)
    campaign.cfg, campaign.run, campaign.data = cfg, tmp_path / "run", tmp_path / "data"
    campaign.config_path, campaign.runtime = str(tmp_path / "config.json"), {}
    campaign.prepare_directions = lambda *a: vector
    calls = []
    campaign.branch_acceptance = lambda v, r: calls.append(("acceptance", r))
    campaign.stage = lambda n, command: calls.append((n, command))
    campaign.hf_stage = lambda n, stage, *a: calls.append((stage, []))
    campaign.cpu_stage = lambda n, command: calls.append(("cpu_teacher", command))
    campaign.report_progress = lambda: None
    campaign.train_step("steer", 1)
    job = read_json(tmp_path / "run/jobs/steer_0001_rollout.json")
    assert job["per_question_vectors"] and not job["decode_cuda_graph"]
    assert all(r.get("vector_id") and r.get("vector_hash") for r in job["requests"])
    assert job["requests"][0]["vector_id"] != job["requests"][1]["vector_id"]
    command = next(command for name, command in calls if name == "steer_0001_rollout")
    assert command[0] == ("scripts/figure8_generate.py" if gate == "all" else "scripts/figure8_generate_controlled.py")
    assert "student" not in [name for name, _ in calls]
    assert ("cpu_teacher" in [name for name, _ in calls]) == (gate == "overlap")
    if gate == "overlap":
        campaign.train_step("vanilla", 1)
        plain = read_json(tmp_path / "run/jobs/vanilla_0001_rollout.json")
        assert plain["backend"] == "hf_cached_causal_gate_reference"
        assert plain["vector"] is None and plain["method"]["gate"] == "all"
        assert all("vector_id" not in r for r in plain["requests"])


def test_diagnostic_supervisor_stops_after_frozen_confirm_results(tmp_path, monkeypatch):
    tuning = module("run_pre_opd_overlap_tuning", monkeypatch)
    monkeypatch.setattr(tuning, "ROOT", tmp_path)
    monkeypatch.setattr(tuning, "relative", lambda path: str(Path(path).relative_to(tmp_path)))
    campaign = tuning.TuningCampaign.__new__(tuning.TuningCampaign)
    campaign.run, campaign.config_path = tmp_path, str(tmp_path / "config.json")
    campaign.cfg = {**read_json(ROOT / "configs/pre_opd_overlap_tuning.json"), "fixed_prefix_source": "source"}
    campaign.prepare_directions = lambda *args: tmp_path / "vectors.pt"
    campaign.train_step = lambda *args: pytest.fail("Diagnostic entered OPD")
    campaign.acceptance = lambda *args: pytest.fail("Diagnostic entered backward acceptance")
    phases = []
    def fake_stage(name, command):
        phases.append(name)
        job = read_json(command[-1])
        summaries = {}
        for c in job["conditions"]:
            score = c.get("rho", 0)
            summaries[c["id"]] = {"condition": c, "question_weighted_delta_pp": score,
                "paired_bootstrap_95ci_pp": [score, score], "token_weighted_delta_pp": score,
                "improved_questions": 1, "questions": 1, "per_question_delta_pp": {"q1": score}}
        atomic_json(tmp_path / job["output"] / "summaries.json", summaries)
    campaign.stage = fake_stage
    campaign.execute_diagnostic()
    assert phases == ["diagnostic_select", "diagnostic_confirm"]
    assert read_json(tmp_path / "campaign.json")["status"] == "awaiting_user_review"
    assert not read_json(tmp_path / "campaign.json")["automatic_opd"]
    assert (tmp_path / "per_question.csv").exists() and (tmp_path / "README.md").exists()


def test_pair_extraction_uses_selected_roles_one_checkpoint_and_rejects_changed_text(tmp_path):
    model = tiny(13)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    class Tokenizer:
        all_special_ids = []
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) % 47 for c in text],
                    "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    question = {"id": "q1", "student_prompt_ids": [1, 2]}
    dirs = {role: tmp_path / role for role in ["teacher", "reference", "student"]}
    for role, text in [("teacher", "abcd"), ("reference", "xyz"), ("student", "lmnop")]:
        atomic_json(dirs[role] / "q1.json", {"id": "q1", "text": text, "finish_reason": "stop"})
    identity = {"checkpoint": "S0", "step": 0}
    differences = {}
    for direction in ["T-S", "T-R", "R-S"]:
        value = extract_pairs(model, Tokenizer(), [question], dirs, Method(direction=direction, layers=(0, 1)),
                              tmp_path / direction, identity)["q1"]
        assert value["usable"]
        differences[direction] = value["layers"]["0"]["pools"][0] - value["layers"]["0"]["pools"][1]
    torch.testing.assert_close(differences["T-S"], differences["T-R"] + differences["R-S"])
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    atomic_json(dirs["teacher"] / "q1.json", {"id": "q1", "text": "changed", "finish_reason": "stop"})
    with pytest.raises(ValueError, match="texts/checkpoint changed"):
        extract_pairs(model, Tokenizer(), [question], dirs, Method(direction="T-R", layers=(0, 1)), tmp_path / "T-R", identity)


def test_branch_full_support_acceptance_is_cpu_testable_and_forbidden_in_zero_update_mode(monkeypatch):
    acceptance = module("figure8_branch_acceptance", monkeypatch)
    student, teacher = tiny(14), tiny(15)
    controller = CompositeSteeringController(student, {0: torch.linspace(-0.2, 0.2, 16)})
    row, score = generate_response(student, teacher, {"prompt_token_ids": [1, 2], "seed": 12}, controller,
              Method(layers=(0,), gate="random", random_probability=0.5), max_tokens=5,
              support_size=47, k=16, stop_ids=set(), audit_full=True)
    assert acceptance.verify_full_replay(student, row, score, controller)["passed"]
    controller.detach()
    with pytest.raises(ValueError, match="forbidden"):
        acceptance.accept({"optimizer_updates": 0}, "not-read.pt")


def test_matched_plain_policy_uses_no_probe_and_records_no_intervention():
    student, teacher = tiny(16), tiny(17)
    calls = []
    handle = student.model.register_forward_pre_hook(lambda *args: calls.append(True))
    row, score = generate_response(student, teacher, {"prompt_token_ids": [1, 2], "seed": 13}, None,
              Method(layers=(0,), gate="all"), max_tokens=4, support_size=47, k=16, stop_ids=set())
    handle.remove()
    assert len(calls) == 4 and row["steering_mask"] == [False] * 4
    assert "probe_topk_ids" not in score
    with torch.inference_mode():
        replay = project_logits(student, response_hidden(student, row)).log_softmax(-1)
    torch.testing.assert_close(replay.gather(-1, score["ids"].long()), score["logps"], rtol=1e-5, atol=1e-5)


def test_first_diagnostic_rejects_a_wrong_direction_bank_before_loading_gpu_model(tmp_path, monkeypatch):
    stage = module("pre_opd_overlap_stage", monkeypatch)
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    cfg = {"student": "S0", "optimizer_updates": 0, "shared_vocabulary_size": 47,
           "steering_method": {"direction": "T-S", "gate": "all", "granularity": "global", "extractor": "initial"}}
    atomic_json(tmp_path / "config.json", cfg)
    wrong = bank(Method(direction="T-R", layers=(0,)))
    wrong["identity"]["checkpoint"] = "S0"
    atomic_torch(tmp_path / "bank.pt", wrong)
    monkeypatch.setattr(stage, "load_model", lambda *a: pytest.fail("Loaded GPU model before checking bank"))
    with pytest.raises(ValueError, match="initial-student global bank"):
        stage.score_job({"config": "config.json", "bank": "bank.pt"})
