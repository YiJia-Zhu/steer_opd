import copy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from steer_opd.steering import SteeringController, prediction_mask
from verl.trainer.distillation.fsdp.losses import tail_aware_kl_divergence


def small_model():
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=47, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config)


def test_prediction_positions_include_last_prompt_state_and_no_wrap():
    response = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]])
    assert prediction_mask(response).tolist() == [[False, True, True, False, False],
                                                [True, True, True, True, False]]


def test_zero_strength_outputs_gradients_and_scope_removal():
    model = small_model().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    baseline = model(tokens).logits
    baseline.sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    controller = SteeringController(model.model.layers[0], torch.randn(32), 0)
    with controller.loss_scope(prediction_mask(torch.tensor([[0, 0, 1, 1, 1]]))):
        observed = model(tokens).logits
        observed.sum().backward()
    assert torch.equal(baseline, observed)
    assert all(torch.equal(grads[n], p.grad) for n, p in model.named_parameters())
    assert not controller.enabled and controller.mask is None
    count = controller.injected_forwards
    assert torch.equal(model(tokens).logits, baseline)
    assert controller.injected_forwards == count
    controller.detach()


def test_checkpoint_backward_recomputes_the_same_steering():
    plain = small_model().train()
    recompute = copy.deepcopy(plain).train()
    recompute.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vector = torch.linspace(-0.1, 0.1, 32)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    mask = prediction_mask(torch.tensor([[0, 0, 1, 1, 1]]))
    controllers = []
    for model in [plain, recompute]:
        controller = SteeringController(model.model.layers[0], vector, 0.3)
        controllers.append(controller)
        with controller.loss_scope(mask):
            loss = model(tokens, use_cache=False).logits.square().mean()
            loss.backward()
    for (_, a), (_, b) in zip(plain.named_parameters(), recompute.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
    assert controllers[1].injected_forwards >= controllers[0].injected_forwards
    for controller in controllers:
        controller.detach()


def test_tail_kl_matches_explicit_coarse_distribution_and_gradient():
    student_logits = torch.tensor([[0.3, -0.2, 1.1, 0.5, -0.6]], requires_grad=True)
    teacher = torch.softmax(torch.tensor([[0.4, 0.7, -0.1, 1.2, -0.4]]), -1)
    ids = teacher.topk(2, -1).indices
    log_student = student_logits.log_softmax(-1).gather(-1, ids)
    log_teacher = teacher.log().gather(-1, ids)
    actual, _, _, _ = tail_aware_kl_divergence(log_student, log_teacher, 1e-6)
    p, q = log_teacher.exp(), log_student.exp()
    p = torch.cat([p, 1 - p.sum(-1, keepdim=True)], -1)
    q = torch.cat([q, 1 - q.sum(-1, keepdim=True)], -1)
    expected = (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.sum(), student_logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), student_logits)[0]
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_r1_native_prompt_and_exact_sampled_tokens_are_preserved():
    from transformers import AutoTokenizer
    from steer_opd.math_data import prompt_ids, TASK_SUFFIX
    from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True,
    )
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") is None
    builder = ContinuousTokenBuilder(tokenizer)
    question = "What is 1+1?"
    prefix = builder.build_initial_tokens([{"role": "user", "content": question + TASK_SUFFIX}])
    assert prefix == prompt_ids(tokenizer, question)
    response = tokenizer.encode(r"We obtain $\boxed{2}$.", add_special_tokens=False) + [tokenizer.eos_token_id]
    merged = builder.merge_assistant_tokens(prefix, response)
    assert merged.token_ids == prefix + response
    assert merged.appended_token_count == len(response)
