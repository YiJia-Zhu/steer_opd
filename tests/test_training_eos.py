"""Sampled EOS survives VERL batching, native scoring and the OPD loss mask."""
import asyncio
from dataclasses import asdict
import math
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch
from transformers import AutoTokenizer

from steer_opd.compass_alignment import ByteLevelCodec
from steer_opd.easysteer_static import fixed_prefix_scores
from steer_opd.generation_policy import PRESENCE_FIELD
from steer_opd.math_data import TASK_SUFFIX
from steer_opd.resumable_core import ROOT, read_json
from steer_opd.static_masks import selector_for_predictions
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.overlap_monitor import row_statistics, summarize_statistics
from steer_opd.training.rollout_seeds import SEED_FIELD
from steer_opd.training.scoring import batch_rows, prepare_request
from steer_opd.training.scoring_request import read_full_request, read_native_request, write_request
from steer_opd.training.verl_adapter import attach_advantages, sampled_policy_loss


def test_eos_training_uses_real_native_ids_and_only_sampled_stop_positions(tmp_path, monkeypatch):
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap
    from verl.workers.rollout.replica import TokenOutput
    from steer_opd.training.verl_agent import TrajectorySeedAgentLoop

    paths = read_json(ROOT/'configs/ablation.json')['tokenizer_paths']
    codecs = {role: ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/path, local_files_only=True))
              for role, path in paths.items()}
    student, teacher = codecs['student'], codecs['teacher']
    answer = 'The answer is 42.\n'
    body = student.encode(answer)
    assert body == [785, 4226, 374, 220, 19, 17, 624]
    teacher_body = teacher.encode(answer)
    assert teacher_body == [760, 4087, 369, 220, 19, 17, 13, 198]
    pad = student.tokenizer.pad_token_id
    assert pad == 151643
    width = len(body) + 2
    capped = [*body, 785, 4226]
    rollouts = [
        ([*body, 151645], 'completed'),
        ([*body, pad], 'completed'),
        (capped, 'length'),
        ([*capped, 151645], 'completed'),  # The agent discards the token beyond its limit.
        ([151645], 'completed'),
    ]
    question = 'What is six times seven?'
    messages = [{'role': 'user', 'content': question + TASK_SUFFIX}]
    worker = object.__new__(AgentLoopWorker)
    worker.tokenizer = student.tokenizer
    worker.rollout_config = NS(prompt_length=128, response_length=width)
    worker.processor = None
    worker.reward_loop_worker_handles = None
    worker.distillation_enabled = False
    config = NS(actor_rollout_ref=NS(rollout=worker.rollout_config))

    async def sample_batch():
        padded = []
        for ids, reason in rollouts:
            async def generate(**kwargs):
                return TokenOutput(token_ids=ids, log_probs=[-6.] * len(ids), stop_reason=reason)

            agent = TrajectorySeedAgentLoop(DictConfigWrap(config), NS(generate=generate),
                student.tokenizer, None, None, DictConfigWrap({'apply_chat_template_kwargs':
                    {'enable_thinking': False}}), hf_model_type='qwen3')
            output = await agent.run({}, raw_prompt=messages, **{SEED_FIELD: 42, PRESENCE_FIELD: 2.})
            padded.append(await worker._agent_loop_postprocess(output, validate=False, raw_prompt=messages))
        return worker._postprocess(padded, {
            'uid': np.array([str(i) for i in range(len(padded))]),
            'extra_info': np.array([{'question_id': 'q', 'question': question}] * len(padded)),
        })

    batch = asyncio.run(sample_batch())
    assert batch.batch['responses'][1, -2:].tolist() == [pad, pad]
    rows = batch_rows(batch)
    assert [row.get('sampled_stop_id') for row in rows] == [151645, 151643, None, None, 151645]
    settings = TrainingLossConfig()
    request = prepare_request({'binding': 'eos', 'training_loss': asdict(settings)}, rows, codecs)
    manifest = write_request(tmp_path, request)
    assert read_full_request(tmp_path, manifest['request_id'], 'eos') == request
    _, native = read_native_request(tmp_path, manifest['request_id'], 'eos', 'teacher')
    assert [r['token_ids'] for r in native] == [
        [*teacher_body, 248046], [*teacher_body, 248044],
        teacher.encode(student.bytes(capped).decode()), teacher.encode(student.bytes(capped).decode()),
        [248046],
    ]
    gates = [True] * len(native[0]['token_ids'])
    positions = selector_for_predictions(native[0], gates, mode='fixed_prefix')['prompt_positions']
    assert positions[-1] == len(native[0]['prompt_token_ids']) + len(teacher_body) - 1

    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=NS))
    top16 = {token: NS(rank=rank, logprob=-math.log(32))
             for rank, token in enumerate(range(50000, 50016), start=1)}

    def score_once(inputs, params, **kwargs):
        assert params.prompt_logprobs == 16 and params.ignore_eos
        outputs = []
        for row, engine_input in zip(native, inputs, strict=True):
            prefix, ids = row['prompt_token_ids'], row['token_ids']
            assert engine_input['prompt_token_ids'] == prefix + ids
            observed = [{**top16, token: NS(rank=17, logprob=-4. if token in teacher.eos else -5.)}
                        for token in ids]
            outputs.append(NS(prompt_token_ids=prefix + ids, prompt_logprobs=[None]*len(prefix) + observed,
                              outputs=[NS(logprobs=[top16])]))
        return outputs

    engine_cfg = NS(model_config=NS(logprobs_mode='raw_logprobs'), cache_config=NS(enable_prefix_caching=False))
    engine = NS(llm_engine=NS(vllm_config=engine_cfg), generate=score_once)
    scores = fixed_prefix_scores(engine, native, [False]*len(native), include_sampled=True)
    replies = [{'alignment': row['alignment'], 'teacher_logps': score['sampled_logps']}
               for row, score in zip(request['rows'], scores, strict=True)]
    assert all(eos not in scores[i]['ids'][-1].tolist() for i, eos in [(0, 248046), (1, 248044)])
    attach_advantages(batch, replies, settings)
    mask = batch.batch['response_mask']
    assert mask[:2, -2:].tolist() == [[1, 0], [1, 0]]
    assert mask[2:4].all()
    assert mask[4].tolist() == [1] + [0]*(width-1)
    advantages = batch.batch['advantages']
    torch.testing.assert_close(advantages[:2, -2:], torch.tensor([[2., 0.], [2., 0.]]))
    current = batch.batch['old_log_probs'].clone().requires_grad_(True)
    count = batch.meta_info['steer_opd_valid_tokens']
    sampled_policy_loss(current, advantages, mask, global_valid_tokens=count).backward()
    torch.testing.assert_close(current.grad[:2, -2:], torch.tensor([[-2/count, 0.], [-2/count, 0.]]))

    def overlap(i):
        student_top16 = {key: value[:1].expand(len(rows[i]['token_ids']), *value.shape[1:])
                         for key, value in scores[i].items() if isinstance(value, torch.Tensor)}
        return row_statistics(rows[i], replies[i]['alignment'], student_top16, scores[i], codecs)

    assert overlap(0)['valid_tokens'] == len(body)
    assert summarize_statistics([overlap(4)])['overlap/coverage'] == 0.
