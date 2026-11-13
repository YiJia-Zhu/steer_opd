"""No student replay: exact candidates, durable resume, teacher overlap and loss."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.generation_runtime import prepare_generation_manifest
from steer_opd.resumable_core import (
    ROOT, atomic_json, atomic_torch, digest, original_advantage, original_token_losses,
    project_logits, read_json, response_loss, score_hidden,
)
from steer_opd.rollout_topk import (
    SOURCE, commit_topk_response, extract_topk, load_rollout_topk, overlap_totals,
)
from steer_opd.replay_validation import verify_rollout_topk_replay


def script(name):
    spec = importlib.util.spec_from_file_location(f"_topk_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entries(values, ranks):
    return {key: SimpleNamespace(logprob=value, rank=rank)
            for (key, value), rank in zip(values, ranks)}


def sample_score():
    return extract_topk([4, 1], [entries([(4, -4), (0, -0.5), (1, -1.5), (2, -2.5)], [5, 1, 2, 3]),
                                entries([(2, -2.5), (1, -1.5), (0, -0.5)], [3, 2, 1])], 3, 7)


def test_extra_sample_is_excluded_even_when_its_rank_ties_topk_boundary():
    score = sample_score()
    assert score["ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert score["sampled_logps"].tolist() == [-4, -1.5]
    tied = extract_topk([6], [entries([(6, -2.5), (0, -0.5), (1, -1.5), (2, -2.5)], [3, 1, 2, 3])], 3, 7)
    assert tied["ids"].tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("steps", [None, [], [entries([(0, -1)], [None])],
                                   [entries([(0, -1), (1, -2)], [1, 3])],
                                   [entries([(0, float('nan')), (1, -2)], [1, 2])],
                                   [entries([(0, -1), (9, -2)], [1, 2])]])
def test_incomplete_nonfinite_or_invalid_candidates_rejected(steps):
    with pytest.raises(ValueError):
        extract_topk([0], steps, 2, 7)


def saved_rollout(tmp_path):
    job = {"model": "frozen", "output": "rollouts", "logprobs": 3, "save_rollout_top_k": 3,
           "logprobs_mode": "raw_logprobs", "support_size": 7,
           "requests": [{"key": "a", "prompt_token_ids": [1, 2], "seed": 11}]}
    job, job_hash, _ = prepare_generation_manifest(tmp_path, job)
    row = {**job["requests"][0], "job_hash": job_hash, "token_ids": [4, 1], "sampled_logps": [-4, -1.5]}
    row = commit_topk_response(tmp_path, row, sample_score())
    return job, row


def test_resume_verifies_tensor_provenance_and_preserves_response_bytes(tmp_path):
    job, row = saved_rollout(tmp_path)
    original = (tmp_path / "a.json").read_bytes()
    assert prepare_generation_manifest(tmp_path, job)[2] == []
    assert (tmp_path / "a.json").read_bytes() == original
    score = load_rollout_topk(tmp_path, row, 3, 7)
    score["logps"][0, 0] -= 0.01
    atomic_torch(tmp_path / row["rollout_topk"], score)
    with pytest.raises(ValueError, match="integrity/provenance"):
        prepare_generation_manifest(tmp_path, job)


def test_interruption_before_response_commit_remains_pending(tmp_path, monkeypatch):
    import steer_opd.rollout_topk as module
    job = {"requests": [{"key": "a"}]}
    prepare_generation_manifest(tmp_path, job)

    def interrupted(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(OSError):
        commit_topk_response(tmp_path, {"key": "a", "token_ids": [4, 1]}, sample_score())
    assert (tmp_path / "topk/a.pt").exists()
    assert not (tmp_path / "a.json").exists()
    assert prepare_generation_manifest(tmp_path, job)[2] == job["requests"]


def test_missing_artifact_and_new_protocol_cannot_silently_reuse_old_responses(tmp_path):
    job, row = saved_rollout(tmp_path)
    (tmp_path / row["rollout_topk"]).unlink()
    with pytest.raises(ValueError, match="Missing or invalid"):
        prepare_generation_manifest(tmp_path, job)
    with pytest.raises(ValueError, match="scientific generation"):
        prepare_generation_manifest(tmp_path, {**job, "save_rollout_top_k": 2})


def test_teacher_candidates_and_own_topk_use_one_projection_per_chunk(monkeypatch):
    import steer_opd.resumable_core as core
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 7, bias=False),
                            config=SimpleNamespace(vocab_size=7), _opd_vocab_size=7)
    hidden = torch.randn(5, 4)
    ids = torch.tensor([[6, 0, 3]] * 5)
    calls = []
    original = core.project_logits

    def record(*args, **kw):
        calls.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(core, "project_logits", record)
    with torch.no_grad():
        score = score_hidden(model, hidden, k=3, candidates=ids, chunk=2, own_topk=True)
    assert len(calls) == 3
    lp = original(model, hidden).log_softmax(-1)
    torch.testing.assert_close(score["logps"], lp.gather(-1, ids))
    torch.testing.assert_close(score["own_logps"], lp.topk(3).values)
    assert torch.equal(score["own_ids"].long(), lp.topk(3).indices)


def test_overlap_counts_positions_and_full_probability_shared_mass():
    s = {"ids": torch.tensor([[0, 1], [2, 3]]), "logps": torch.tensor([[0.4, 0.3], [0.5, 0.1]]).log()}
    t = {"ids": torch.tensor([[1, 4], [3, 2]]), "logps": torch.tensor([[0.6, 0.2], [0.4, 0.3]]).log()}
    result = overlap_totals(s, t)
    assert result["positions"] == 2 and result["intersection"] == 3
    assert result["student_shared_mass_sum"] == pytest.approx(0.9)
    assert result["teacher_shared_mass_sum"] == pytest.approx(1.3)


def test_observed_replay_check_detects_wrong_candidate_or_tail_mass():
    p = torch.tensor([[0.5, 0.3]]).log()
    result = verify_rollout_topk_replay(p, p)
    assert result["passed"] and not result["full_vocabulary_kl"]
    for q in [torch.tensor([[0.3, 0.5]]).log(), torch.tensor([[0.1, 0.06]]).log()]:
        with pytest.raises(RuntimeError, match="Top-K/HF replay mismatch"):
            verify_rollout_topk_replay(p, q)


def test_learning_forward_uses_saved_rollout_reward_and_exact_original_surrogate_gradient():
    torch.manual_seed(20)
    cfg = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).eval()
    other = copy.deepcopy(model)
    row = {"prompt_token_ids": [1, 2], "token_ids": [3, 4, 5]}
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hidden = model.model(inputs, use_cache=False).last_hidden_state[0, 1:]
        lp = project_logits(model, hidden).log_softmax(-1)
        values, selected = lp.topk(3)
        # Backend difference is small but nonzero: stored values must be used.
        old = values - 0.001
        teacher = torch.randn(3, 13).log_softmax(-1).gather(-1, selected)
    scores = {"source": SOURCE, "ids": selected.int(), "logps": old, "teacher_logps": teacher}
    loss, error = response_loss(model, row, scores, None, 3, chunk=2)
    current = project_logits(other, other.model(inputs, use_cache=False).last_hidden_state[0, 1:]).log_softmax(-1).gather(-1, selected)
    expected = original_token_losses(current, current.detach(), original_advantage(old, teacher)).mean()
    expected.backward()
    assert loss == pytest.approx(float(expected.detach()))
    assert scores["replay_validation"]["passed"] and error > 0
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_campaign_routes_rollout_directly_to_teacher_then_training(tmp_path, monkeypatch):
    module = script("run_figure8_campaign")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    atomic_json(tmp_path / "data/opd.json", [{"id": "q1", "answer": "1", "student_prompt_ids": [1, 2]}])
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run, campaign.data = tmp_path / "run", tmp_path / "data"
    campaign.cfg = {"student_topk_source": "rollout", "prompt_batch_size": 1, "rollouts_per_prompt": 1,
                    "seed": 42, "max_response_tokens": 5, "student": "model", "supervision_top_k": 16,
                    "vllm_gpu_memory_utilization": 0.75, "shared_vocabulary_size": 151665}
    campaign.runtime = {}
    calls = []
    campaign.stage = lambda name, command: calls.append((name, command))
    campaign.hf_stage = lambda name, stage, *args: calls.append((stage, args))
    campaign.report_progress = lambda: None
    campaign.train_step("vanilla", 1)
    assert [name for name, _ in calls] == ["vanilla_0001_rollout", "teacher", "train"]
    job = read_json(campaign.run / "jobs/vanilla_0001_rollout.json")
    assert job["logprobs"] == 16 and job["save_rollout_top_k"] == 16
    assert job["logprobs_mode"] == "raw_logprobs" and "max_num_seqs" not in job


def test_teacher_stage_consumes_rollout_artifact_and_resumes_overlap_without_student_forward(tmp_path, monkeypatch):
    module = script("figure8_stage")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rollout = tmp_path / "vanilla/steps/0001/rollouts"
    _, row = saved_rollout(rollout)
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 7, bias=False),
                            config=SimpleNamespace(vocab_size=7), _opd_vocab_size=7)
    monkeypatch.setattr(module, "load_model", lambda *args: model)
    monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: range(7)))
    forwards = []

    def forward(*args):
        forwards.append(True)
        return torch.ones(2, 4)

    monkeypatch.setattr(module, "response_hidden", forward)
    cfg = {"run_dir": str(tmp_path), "teacher": "teacher", "student_topk_source": "rollout",
           "supervision_top_k": 3, "shared_vocabulary_size": 7, "logit_chunk_size": 128,
           "input_protocol": "exact_student_token_ids"}
    module.teacher_scores(cfg, "vanilla", 1)
    assert len(forwards) == 1
    stage = tmp_path / "vanilla/steps/0001"
    score = torch.load(stage / "scores/a.pt", weights_only=True)
    assert score["source"] == SOURCE and score["row_hash"] == digest(row)
    assert not (stage / "student_scores").exists()
    expected = read_json(stage / "rollout_overlap.json")
    assert expected["positions"] == 2 and not expected["primary_nohook_dev_metric"]
    (stage / "scores/complete.json").unlink()
    module.teacher_scores(cfg, "vanilla", 1)
    assert len(forwards) == 1
    assert read_json(stage / "rollout_overlap.json") == expected


@pytest.mark.parametrize("hold,state", [(True, {}), (False, {"status": "paused_by_user"})])
def test_paused_campaign_rejects_automatic_restart_before_any_stage(tmp_path, hold, state):
    module = script("run_figure8_campaign")
    cfg = {"run_dir": str(tmp_path), "execution_hold": hold}
    atomic_json(tmp_path / "config_request.json", cfg)
    atomic_json(tmp_path / "campaign.json", state)
    with pytest.raises(ValueError, match="paused by user"):
        module.Campaign(tmp_path / "config_request.json")
    assert read_json(tmp_path / "campaign.json") == state


def test_shared_pool_preserves_exact_original_ids_prompts_and_dev_test_isolation(tmp_path):
    module = script("prepare_figure8_first_run")
    splits = {"opd": [{"id": "q1", "question": "train one", "student_prompt_ids": [1, 2]}],
              "dev": [{"id": "q2", "question": "dev two", "student_prompt_ids": [3, 4]}],
              "aime2025": [{"id": f"a{i}", "question": f"test {i}"} for i in range(30)]}
    manifest = {"source": "frozen.parquet", "splits": {name: {"hash": digest(rows)} for name, rows in splits.items()}}
    for name, rows in splits.items():
        atomic_json(tmp_path / f"{name}.json", rows)
    atomic_json(tmp_path / "manifest.json", manifest)
    cfg = {"source": "frozen.parquet", "train_questions": 1, "direction_questions": 1, "dev_questions": 1}
    reused, _ = module.same_as_opd_splits(tmp_path, cfg)
    assert reused["direction"] == reused["opd"] == splits["opd"]
    assert reused["dev"] == splits["dev"] and reused["aime2025"] == splits["aime2025"]
    splits["dev"][0]["question"] = " TRAIN  ONE "
    atomic_json(tmp_path / "dev.json", splits["dev"])
    manifest["splits"]["dev"]["hash"] = digest(splits["dev"])
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="opd/dev overlap"):
        module.same_as_opd_splits(tmp_path, cfg)
