"""Static causal replay, Slerp math and actual pinned EasySteer API contracts."""

import copy
import importlib
import math
import sys
import types

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.direction_bank import bank_vectors, make_bank
from steer_opd.method_config import Method
from steer_opd.operators import slerp
from steer_opd.resumable_core import ROOT, response_hidden, score_hidden
from steer_opd.static_masks import freeze_static_mask, native_overlap, selector_for_predictions, validate_static_mask
from steer_opd.static_screen import score_batch, score_native_candidate
from steer_opd.steering import SteeringController, response_prediction_mask


def tiny(seed=1):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def rows():
    return [{"id": "q1", "key": "r1", "prompt_token_ids": [1, 2, 3], "token_ids": [4, 5, 6, 0], "steering": False},
            {"id": "q2", "key": "r2", "prompt_token_ids": [1, 7], "token_ids": [8, 9], "steering": False}]


def identity():
    return {"student_checkpoint": "S0", "teacher_checkpoint": "T", "student_tokenizer_sha256": "same",
            "teacher_tokenizer_sha256": "same", "support_id": "47", "metric": "native_overlap_at_16",
            "student_score_sha256": "student", "teacher_score_sha256": "teacher"}


def bank(direction="T-S", granularity="global"):
    method = Method(direction=direction, granularity=granularity, layers=(2,), operator="slerp", alpha=.1)
    pooled = {qid: {"usable": True, "layers": {"2": {"pools": [torch.arange(16).float()+i, torch.arange(16).flip(0).float()],
               "norms": [torch.tensor([10., 12.]), torch.tensor([11.])]}}} for i, qid in enumerate(["q1", "q2"], 1)}
    return make_bank(pooled, method, {"checkpoint": "S0", "step": 0})


@pytest.mark.parametrize("alpha", [.01, .1, .3, .5, 1.])
def test_slerp_angle_norm_and_direction_scale(alpha):
    h = torch.tensor([[3., 0., 0.]], dtype=torch.float64)
    v = torch.tensor([.5, math.sqrt(3)/2, 0.], dtype=torch.float64)
    changed = slerp(h, v, alpha)
    expected = torch.tensor([[3*math.cos(alpha*math.pi/3), 3*math.sin(alpha*math.pi/3), 0.]], dtype=torch.float64)
    torch.testing.assert_close(changed, expected)
    torch.testing.assert_close(changed.norm(dim=-1), h.norm(dim=-1))
    torch.testing.assert_close(changed, slerp(h, v*1000, alpha))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slerp_zero_parallel_antiparallel_and_backward(dtype):
    h = torch.tensor([[1., 0.], [-1., 0.], [0., 0.], [0., 1.]], dtype=dtype, requires_grad=True)
    v = torch.tensor([1., 0.], dtype=dtype)
    assert slerp(h, v, 0) is h
    assert torch.equal(slerp(h, v*0, .1), h)
    result = slerp(h, v, .1)
    assert torch.equal(result[[0, 2]], h[[0, 2]])
    torch.testing.assert_close(result.float().norm(dim=-1), h.float().norm(dim=-1), rtol=.005, atol=.005)
    assert result[1, 1] > 0  # deterministic antipodal great circle
    result.float().square().sum().backward()
    assert torch.isfinite(h.grad).all()


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_slerp_strength_rejected(alpha):
    with pytest.raises(ValueError):
        slerp(torch.ones(2), torch.ones(2), alpha)


def test_method_rejects_removed_dynamic_gate_and_binds_static_direction_identity():
    with pytest.raises(ValueError,match='all/static_overlap'):
        Method(gate='overlap',overlap_threshold=.5)
    method = Method(gate="static_overlap", overlap_threshold=.5, operator="slerp", alpha=.1)
    assert method.to_dict()["alpha"] == .1
    value = bank(granularity="per_question")
    vectors, first = bank_vectors(value, "q1", alpha=.1, layers=[2])
    assert vectors[2].norm() == pytest.approx(1)
    assert first != bank_vectors(value, "q1", alpha=.3, layers=[2])[1]
    with pytest.raises(ValueError, match="No usable"):
        bank_vectors(value, "missing")


def test_threshold_exact_precision_and_mask_source_binding():
    row = rows()[0]
    exact = freeze_static_mask(row, [0, .375, .5, 1], .375, identity())
    assert exact["requested_mask"] == [True, False, False, False]
    adjacent = freeze_static_mask(row, [0, .375, .5, 1], math.nextafter(.375, math.inf), identity())
    assert adjacent["requested_mask"] == [True, True, False, False]
    assert validate_static_mask(exact, row) == exact["requested_mask"]
    changed = copy.deepcopy(row); changed["token_ids"][0] = 10
    with pytest.raises(ValueError, match="changed"):
        validate_static_mask(exact, changed)
    changed_artifact = copy.deepcopy(exact); changed_artifact["requested_mask"][1] = True
    with pytest.raises(ValueError, match="changed"):
        validate_static_mask(changed_artifact, row)
    with pytest.raises(ValueError, match="changed"):
        validate_static_mask(exact, row, {**identity(), "student_checkpoint": "S1"})


def test_controlled_history_and_cross_tokenizer_native_ids_rejected():
    row = {**rows()[0], "steering": True}
    with pytest.raises(ValueError, match="controlled"):
        freeze_static_mask(row, [0]*4, .5, identity())
    with pytest.raises(ValueError, match="identical tokenizer"):
        freeze_static_mask(rows()[0], [0]*4, .5, {**identity(), "teacher_tokenizer_sha256": "other"})
    with pytest.raises(ValueError, match="Cross-tokenizer"):
        native_overlap(torch.arange(16)[None], torch.arange(16)[None], {**identity(), "teacher_tokenizer_sha256": "other"})


def test_prediction_selectors_include_first_eos_and_do_not_wrap():
    row = rows()[0]; gates = [True, False, True, True]
    assert selector_for_predictions(row, gates, mode="fixed_prefix") == {"prompt_positions": [2, 4, 5]}
    assert selector_for_predictions(row, gates, mode="second_generation") == {"prompt_positions": [-1], "generation_positions": [1, 2]}
    assert selector_for_predictions(row, [False]*4, mode="fixed_prefix") is None
    assert selector_for_predictions(row, [False, True, False, False], mode="second_generation") == {"generation_positions": [0]}
    assert response_prediction_mask(3, 4, gates).tolist() == [[False, False, True, False, True, True]]


def test_batched_slerp_matches_sequential_per_question_masks_and_padding():
    model = tiny(); source = rows()
    values = [{2: torch.linspace(-1, 1, 16)}, {2: torch.linspace(1, -1, 16)}]
    gates = [[True, False, True, True], [False, True]]
    actual = score_batch(model, source, vectors=values, gates=gates, alpha=.1)
    for row, vec, mask, score in zip(source, values, gates, actual):
        controller = SteeringController(model.model.layers[2], vec[2], .1, operator="slerp")
        with torch.inference_mode():
            expected = score_hidden(model, response_hidden(model, {**row, "steering_mask": mask}, controller))
        controller.detach()
        assert torch.equal(score["ids"], expected["ids"])
        torch.testing.assert_close(score["logps"], expected["logps"], atol=1e-6, rtol=1e-6)
        ratio = score["relative_perturbation"]["2"]
        assert torch.equal(ratio[~torch.tensor(mask)], torch.zeros(sum(not g for g in mask)))
        assert (ratio[torch.tensor(mask)] > 0).all()
    assert not model.model.layers[2]._forward_hooks


def test_all_eight_static_candidates_no_teacher_forward_and_no_mask_redecision():
    student, teacher = tiny(3), tiny(4)
    source = rows()
    teacher_records = {r["key"]: s for r, s in zip(source, score_batch(teacher, source))}
    masks = {r["key"]: freeze_static_mask(r, [.25]*len(r["token_ids"]), .5, identity()) for r in source}
    original_masks = copy.deepcopy(masks)
    baseline = score_native_candidate(student, source, teacher_records, masks, bank(), None)
    conditions = [(d, g, v) for d in ["T-S", "T-R"] for g in ["all", "static_overlap"] for v in ["global", "per_question"]]
    for direction, gate, granularity in conditions:
        condition = {"direction": direction, "gate": gate, "granularity": granularity,
                     "layers": [2], "operator": "slerp", "alpha": .1}
        result = score_native_candidate(student, source, teacher_records, masks, bank(direction, granularity), condition, baseline=baseline)
        assert all(r["parameter_updates"] == 0 and not r["current_step_probe"] for r in result.values())
        assert all(r["steering_mask"].all() for r in result.values())
        zero = score_native_candidate(student, source, teacher_records, masks, bank(direction, granularity), {**condition, "alpha": 0})
        assert all(torch.equal(zero[key]["logps"], baseline[key]["logps"]) for key in baseline)
        assert all(not r["steering_mask"].any() for r in zero.values())
    assert masks == original_masks




@pytest.fixture
def upstream_api(monkeypatch):
    """Load actual pinned source API/math on CPU, without importing a GPU engine."""
    root = ROOT / "ref/easysteer_runtime"
    before = dict(sys.modules)
    for name in ("vllm", "vllm.model_hooks", "vllm.model_hooks.steering", "vllm.model_hooks.steering.algorithms", "vllm.model_hooks.selection"):
        module = types.ModuleType(name)
        module.__path__ = [str(root / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, module)
    try:
        yield importlib.import_module("vllm.model_hooks.steering.api")
    finally:
        for name in list(sys.modules):
            if name.startswith("vllm.") and name not in before:
                del sys.modules[name]


def test_real_easysteer_authoring_contract_plugin_and_empty_selection(upstream_api):
    from steer_opd.easysteer_static import build_spec
    from steer_opd.easysteer_plugin import ALGORITHM, register
    assert register() and register()
    value = build_spec(rows()[0], [True, False, True, True], {2: torch.arange(16).float()})
    vector = value.vectors[0]
    assert vector.algorithm == ALGORITHM and vector.scale == .1
    assert vector.apply.prompt_positions == [2, 4, 5]
    from vllm.model_hooks.steering.algorithms.registry import ALGORITHM_REGISTRY
    algo = ALGORITHM_REGISTRY[ALGORITHM]()
    algo.set_payload(torch.tensor([0., 1.]), .1)
    torch.testing.assert_close(algo._transform(torch.tensor([[1., 0.]]), algo.payload),
                               torch.tensor([[math.cos(math.pi/20), math.sin(math.pi/20)]]))
    assert algo.graph_family is None
    assert build_spec(rows()[0], [False]*4, {2: torch.ones(16)}) is False
    assert build_spec(rows()[0], [True]*4, {2: torch.zeros(16)}) is False
    value = build_spec(rows()[0], [True, False, True, True], {2: torch.ones(16)}, mode="second_generation")
    assert value.vectors[0].apply.generation_positions == [1, 2]


def test_real_upstream_selector_matches_fixed_prefix_and_generation(upstream_api):
    import numpy as np
    from vllm.model_hooks.selection.host import clause_mask
    row = rows()[0]; gates = [True, False, True, True]
    for mode, prompt_len in [("fixed_prefix", 6), ("second_generation", 3)]:
        abs_pos = np.arange(9)
        is_dec = abs_pos >= prompt_len
        gen_idx = abs_pos-prompt_len
        clause = selector_for_predictions(row, gates, mode=mode)
        selected = clause_mask(clause, is_dec, abs_pos, np.full(9, prompt_len), gen_idx, lambda: np.zeros(9))
        assert np.flatnonzero(selected).tolist() == [2, 4, 5]


def screen_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_static_overlap_screen")


def job_fixture(tmp_path, monkeypatch):
    from steer_opd.resumable_core import atomic_json, atomic_torch, digest
    module = screen_module(monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = rows()
    student_scores, teacher_scores = score_batch(tiny(3), source), score_batch(tiny(4), source)
    for row, a, b in zip(source, student_scores, teacher_scores):
        key = row["key"]
        source_hash = digest({"prompt_token_ids": row["prompt_token_ids"], "token_ids": row["token_ids"]})
        atomic_torch(tmp_path / "student_scores" / f"{key}.pt", {**a, "source": "vllm_rollout_raw_logprobs", "support_size": 47, "source_token_hash": source_hash})
        atomic_torch(tmp_path / "teacher_scores" / f"{key}.pt", {**b, "support_size": 47, "source_token_hash": source_hash, "teacher_checkpoint": "T"})
    atomic_json(tmp_path / "rows.json", source)
    banks, conditions = {}, []
    for direction in ["T-S", "T-R"]:
        for granularity in ["global", "per_question"]:
            name = f"{direction}_{granularity}"
            banks[name] = f"banks/{name}.pt"
            atomic_torch(tmp_path / banks[name], bank(direction, granularity))
            for gate in ["all", "static_overlap"]:
                conditions.append({"id": f"{name}_{gate}", "bank": name, "direction": direction, "granularity": granularity,
                                   "gate": gate, "layers": [2], "operator": "slerp", "alpha": .1})
    job = {"stage": "static_overlap_screen", "optimizer_updates": 0, "student": "S0", "identity": identity(),
           "rows": "rows.json", "banks": banks, "conditions": conditions, "student_scores": "student_scores",
           "teacher_scores": "teacher_scores", "threshold": .5, "support_size": 47, "output": "result"}
    return module, job


def test_cli_cpu_prepare_all_eight_and_changed_input_rejection(tmp_path, monkeypatch):
    module, job = job_fixture(tmp_path, monkeypatch)
    result = module.prepare(job)
    assert len(result[1]) == 2 and len(result[3]) == 4
    assert len(list((tmp_path / "result/masks").glob("*.json"))) == 2
    assert module.prepare(job)[1] == result[1]
    with pytest.raises(ValueError, match="Cannot resume"):
        module.prepare({**job, "threshold": .6})
    with pytest.raises(ValueError, match="Cross-tokenizer"):
        module.prepare({**job, "identity": {**identity(), "teacher_tokenizer_sha256": "different"}})


def test_cli_zero_update_scoring_commits_plots_and_resumes_without_model(tmp_path, monkeypatch):
    from steer_opd import resumable_core
    module, job = job_fixture(tmp_path, monkeypatch)
    job["conditions"] = job["conditions"][:1]
    prepared = module.prepare(job)
    model = tiny(3)
    monkeypatch.setattr(resumable_core, "require_gpu", lambda: None)
    monkeypatch.setattr(resumable_core, "load_model", lambda _: model)
    module.run(job, *prepared)
    from steer_opd.resumable_core import read_json
    assert read_json(tmp_path / "result/complete.json")["parameter_updates"] == 0
    condition = job["conditions"][0]["id"]
    assert (tmp_path / "result" / condition / "comparison/first3.pdf").exists()
    def forbid(*args):
        raise AssertionError("A complete cached screen must not load a model or require a GPU")
    monkeypatch.setattr(resumable_core, "require_gpu", forbid)
    monkeypatch.setattr(resumable_core, "load_model", forbid)
    module.run(job, *prepared)
    target = tmp_path / "result" / condition / "r1.pt"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        module.run(job, *prepared)


def test_fixed_prefix_engine_logprob_alignment_including_single_eos(upstream_api, monkeypatch):
    from types import SimpleNamespace as N
    from steer_opd.easysteer_static import fixed_prefix_inputs, fixed_prefix_scores
    import vllm
    monkeypatch.setattr(vllm, "SamplingParams", lambda **kw: N(**kw), raising=False)
    source = [rows()[0], {"id": "q2", "prompt_token_ids": [1, 2], "token_ids": [0]}]
    def entries(offset):
        return {offset+i: N(rank=i+1, logprob=-3.-i*.1) for i in range(16)}
    outputs = []
    for row in source:
        p, n = len(row["prompt_token_ids"]), len(row["token_ids"])
        prompt = row["prompt_token_ids"]+row["token_ids"][:-1]
        logps = [None]+[entries(100+j) for j in range(1, len(prompt))]
        outputs.append(N(prompt_token_ids=prompt, prompt_logprobs=logps,
                         outputs=[N(logprobs=[entries(200)])]))
    class Engine:
        llm_engine = N(vllm_config=N(model_config=N(logprobs_mode="raw_logprobs"), cache_config=N(enable_prefix_caching=False)))
        def generate(self, inputs, params, *, steering, use_tqdm):
            assert inputs == fixed_prefix_inputs(source) and steering == [False, False]
            assert params.max_tokens == 1 and params.prompt_logprobs == 16
            return outputs
    records = fixed_prefix_scores(Engine(), source, [False, False])
    assert records[0]["ids"][:, 0].tolist() == [103, 104, 105, 200]
    assert records[1]["ids"][:, 0].tolist() == [200]


def test_easysteer_admission_does_not_depend_on_the_separate_opd_version_pin(monkeypatch):
    from steer_opd import easysteer_static
    monkeypatch.setattr(easysteer_static, "require_easysteer", lambda: None)
    monkeypatch.delenv("STEER_OPD_GPU_ADMISSION", raising=False)
    with pytest.raises(RuntimeError, match="admission"):
        easysteer_static.require_easysteer_gpu()
    monkeypatch.setenv("STEER_OPD_GPU_ADMISSION", "test")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="exactly one"):
        easysteer_static.require_easysteer_gpu()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    easysteer_static.require_easysteer_gpu()


def test_full_prompt_scoring_keeps_observed_eos_outside_top16_rank_tie(upstream_api, monkeypatch):
    from types import SimpleNamespace as N
    from steer_opd.easysteer_static import fixed_prefix_inputs, fixed_prefix_scores
    import vllm
    monkeypatch.setattr(vllm, "SamplingParams", lambda **kw: N(**kw), raising=False)
    row = {"id": "q", "prompt_token_ids": [1, 2], "token_ids": [0]}
    entries = {i+10: N(rank=i+1, logprob=-3.-i*.1) for i in range(16)}
    entries[0] = N(rank=16, logprob=-4.5)  # extra observed EOS, tied with the boundary
    class Engine:
        llm_engine = N(vllm_config=N(model_config=N(logprobs_mode="raw_logprobs"), cache_config=N(enable_prefix_caching=False)))
        def generate(self, inputs, params, *, steering, use_tqdm):
            assert inputs == fixed_prefix_inputs([row], include_sampled=True)
            return [N(prompt_token_ids=[1, 2, 0], prompt_logprobs=[None, {}, entries], outputs=[N(logprobs=[entries])])]
    result = fixed_prefix_scores(Engine(), [row], [False], include_sampled=True)[0]
    assert result["ids"].tolist() == [list(range(10, 26))]
    assert result["sampled_logps"].tolist() == [-4.5]
