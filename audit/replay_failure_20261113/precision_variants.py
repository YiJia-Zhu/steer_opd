"""Compare computation variants on frozen trajectories, without learning."""
import os
from contextlib import nullcontext
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations import sdpa_attention
from diagnose import comparison
from steer_opd.resumable_core import (ROOT, atomic_json, controller_for, load_model,
                                     read_json, response_hidden, score_hidden, emit)
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.steering import response_prediction_mask

job = read_json(ROOT / 'runs/full_scheme_20261112/learning_selection/tr_l20_6_rho0p01/jobs/steer_0001_rollout.json')
os.environ['STEER_OPD_SUPPORT_SIZE'] = str(job['support_size'])
model = load_model(ROOT / job['model'])
original_gqa = sdpa_attention.use_gqa_in_sdpa
reports = []
with torch.inference_mode():
    for mode in ['math_full', 'repeat_kv', 'explicit_causal_mask', 'fp32']:
        if mode == 'fp32':
            model.float()
        sdpa_attention.use_gqa_in_sdpa = (lambda *args: False) if mode == 'repeat_kv' else original_gqa
        for key in ['018_0', '024_1', '029_0', '033_3', '062_2']:
            row = read_json(ROOT / job['output'] / f'{key}.json')
            old = load_rollout_topk(ROOT / job['output'], row, 16, job['support_size'])
            controller = controller_for(model, ROOT / row['vector_path'], row)
            context = sdpa_kernel(SDPBackend.MATH) if mode == 'math_full' else nullcontext()
            try:
                with context:
                    if mode == 'explicit_causal_mask':
                        tokens = torch.tensor([row['prompt_token_ids'] + row['token_ids'][:-1]], device='cuda')
                        mask = response_prediction_mask(len(row['prompt_token_ids']), len(row['token_ids']),
                                                       row['steering_mask'], device='cuda')
                        attention = torch.ones((1, 1, tokens.shape[1], tokens.shape[1]), dtype=torch.bool, device='cuda').tril()
                        with controller.loss_scope(mask):
                            hidden = model.model(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state[0, len(row['prompt_token_ids'])-1:]
                    else:
                        hidden = response_hidden(model, row, controller)
                    new = score_hidden(model, hidden, candidates=old['ids'])
                report = {'key': key, 'mode': mode, **comparison(old['logps'], new['logps'])}
                reports.append(report)
                emit('precision_variant', key=key, mode=mode, passed=report['passed'],
                     max_kl=max(v for k,v in report.items() if k.endswith('_max_kl') and not k.startswith('threshold')))
                atomic_json(ROOT / 'audit/replay_failure_20261113/precision_variants.json', {'experimental_updates': 0, 'reports': reports})
            finally:
                controller.detach()
sdpa_attention.use_gqa_in_sdpa = original_gqa
