"""Actual failure modes: fork gradient, historical hooks, and optimizer resume."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.resumable_core import FP32OutputLinear, MasterAdam, atomic_torch, original_advantage, original_token_losses, project_logits
from steer_opd.steering import SteeringController


def extracted(path, names, namespace):
    selected = [n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in selected:
        node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def test_reference_surrogate_values_gradients_and_microbatch_accumulation():
    root = Path(__file__).resolve().parents[1] / "ref/ref_rep/Rethinking-OPD/verl/verl"
    ns = {"torch": torch, "AlgoConfig": type("Unused", (), {})}
    extracted(root / "utils/torch_functional.py", ["masked_mean", "masked_sum"], ns)
    ns["verl_F"] = SimpleNamespace(masked_mean=ns["masked_mean"])
    extracted(root / "trainer/ppo/core_algos.py", ["agg_loss", "compute_policy_loss_vanilla"], ns)
    extracted(root / "workers/actor/dp_actor.py", ["compute_reward_weights"], ns)
    torch.manual_seed(17)
    old = torch.randn(2, 5, 4, dtype=torch.float64).log_softmax(-1)
    teacher = torch.randn_like(old).log_softmax(-1)
    reference_adv = (teacher - old) * ns["compute_reward_weights"](
        old, teacher, torch.ones_like(old, dtype=torch.bool), "student_p")
    torch.testing.assert_close(original_advantage(old, teacher), reference_adv)
    # Nonzero policy changes exercise clipping, both signs and dual clipping.
    a = (old + torch.randn_like(old) * 0.8).requires_grad_()
    b = a.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.float64)
    cfg = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                           "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    expected, _ = ns["compute_policy_loss_vanilla"](old, a, reference_adv, mask, config=cfg)
    actual = (original_token_losses(b, old, original_advantage(old, teacher)) * mask).sum() / (mask.sum() + 1e-8)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(a.grad, b.grad)


def tiny_model():
    cfg = Qwen3Config(vocab_size=47, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def test_cached_controlled_decode_matches_full_replay_and_checkpoint_gradients():
    torch.manual_seed(3)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    controller = SteeringController(model.model.layers[1], torch.linspace(-0.1, 0.1, 32), 0.3)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 2:] = True
    with torch.no_grad(), controller.loss_scope(mask):
        full = model(ids, use_cache=False).logits
    cache, replay = None, []
    with torch.no_grad():
        for end in range(3, 8):
            tokens = ids[:, :end] if cache is None else ids[:, end - 1:end]
            current_mask = torch.zeros_like(tokens, dtype=torch.bool)
            current_mask[:, -1] = True
            with controller.loss_scope(current_mask):
                out = model(tokens, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            replay.append(out.logits[:, -1])
    torch.testing.assert_close(torch.stack(replay, 1), full[:, 2:], rtol=2e-5, atol=2e-6)
    controller.detach()
    other = copy.deepcopy(model)
    other.train().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for current in [model, other]:
        hook = SteeringController(current.model.layers[1], torch.linspace(-0.1, 0.1, 32), 0.3)
        with hook.loss_scope(mask):
            current(ids, use_cache=False).logits.square().mean().backward()
        hook.detach()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_vllm_residual_recombination_matches_hf_and_preserves_unmasked_rows():
    torch.manual_seed(9)
    mlp, residual, delta = (torch.randn(4, 32).bfloat16() for _ in range(3))
    mask = torch.tensor([False, True, False, True])[:, None]
    combined = (mlp + residual) + delta
    new_hidden = torch.where(mask, combined, mlp)
    new_residual = torch.where(mask, 0, residual)
    assert torch.equal(new_hidden + new_residual, torch.where(mask, combined, mlp + residual))
    assert torch.equal(new_hidden[~mask[:, 0]], mlp[~mask[:, 0]])


def test_adam_checkpoint_restores_next_update_exactly(tmp_path):
    # Same master-parameter Adam state format, verified across serialization.
    a = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32))
    opt_a = torch.optim.AdamW([a], lr=1e-6, weight_decay=0.01)
    a.grad = torch.tensor([0.3, -0.1])
    opt_a.step()
    torch.save({"master": a.detach().clone(), "optimizer": opt_a.state_dict()}, tmp_path / "resume.pt")
    saved = torch.load(tmp_path / "resume.pt", weights_only=True)
    b = torch.nn.Parameter(saved["master"])
    opt_b = torch.optim.AdamW([b], lr=1e-6, weight_decay=0.01)
    opt_b.load_state_dict(saved["optimizer"])
    for p, opt in [(a, opt_a), (b, opt_b)]:
        p.grad = torch.tensor([-0.2, 0.5])
        opt.step()
    assert torch.equal(a, b)


def test_real_master_optimizer_resume_keeps_bf16_weights_and_fp32_accumulation(tmp_path):
    torch.manual_seed(8)
    first = torch.nn.Linear(4, 3).bfloat16()
    config = {"learning_rate": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
    opt_first = MasterAdam(first, config)
    x = torch.randn(2, 4).bfloat16()

    def update(model, optimizer):
        for row in x:
            model(row).float().square().sum().backward()
            optimizer.accumulate()
        assert all(p.grad.dtype == torch.float32 for p in optimizer.master)
        optimizer.step(1.0)

    update(first, opt_first)
    atomic_torch(tmp_path / "resume.pt", opt_first.state_dict())
    second = torch.nn.Linear(4, 3).bfloat16()
    opt_second = MasterAdam(second, config)
    opt_second.load(torch.load(tmp_path / "resume.pt", weights_only=False))
    update(first, opt_first)
    update(second, opt_second)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(opt_first.master, opt_second.master):
        assert torch.equal(a, b)


def test_fp32_output_linear_backward_matches_float_reference():
    torch.manual_seed(4)
    x = torch.randn(3, 8).bfloat16().requires_grad_()
    w = torch.randn(13, 8).bfloat16().requires_grad_()
    a, b = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    output = FP32OutputLinear.apply(x, w)
    reference = torch.nn.functional.linear(a.float(), b.float())
    torch.testing.assert_close(output, reference)
    output.square().mean().backward()
    reference.square().mean().backward()
    assert torch.equal(x.grad, a.grad)
    assert torch.equal(w.grad, b.grad)


def test_undefined_output_rows_cannot_enter_shared_distribution_or_head_gradient():
    head = torch.nn.Linear(4, 7, bias=False).bfloat16()
    with torch.no_grad():
        head.weight.fill_(0.1)
        head.weight[5:].fill_(20)
    model = SimpleNamespace(lm_head=head, _opd_vocab_size=5)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    assert project_logits(model, hidden, full_vocab=True).argmax(-1).min() >= 5
    logits = project_logits(model, hidden)
    assert logits.shape == (2, 5)
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(2))
    logits.log_softmax(-1)[:, 0].sum().backward()
    assert torch.equal(head.weight.grad[5:], torch.zeros_like(head.weight.grad[5:]))


def test_direction_capture_is_identical_when_unused_later_blocks_are_skipped():
    torch.manual_seed(12)
    full = tiny_model()
    short = copy.deepcopy(full)
    short.model.layers = torch.nn.ModuleList(list(short.model.layers[:2]))
    captured = []
    for model in [full, short]:
        handle = model.model.layers[1].register_forward_hook(lambda module, args, value: captured.append(value.clone()))
        with torch.inference_mode():
            model.model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
        handle.remove()
    assert torch.equal(captured[0], captured[1])
