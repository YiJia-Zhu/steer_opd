"""Recompute a global teacher-space S-T direction from each training batch."""
from collections import defaultdict
import math

import torch

from steer_opd.easysteer_extraction import capture_budgets
from steer_opd.extraction import encode_answer
from steer_opd.math_data import prompt_text


PROTOCOL = 'teacher_space_student_minus_teacher_reference_per_step_v1'


def group_rollouts(rows, *, batch_size, rollout_n):
    groups = defaultdict(list)
    for row in rows:
        groups[row['question_id']].append(row)
    if len(groups) != batch_size or any(len(group) != rollout_n for group in groups.values()):
        raise ValueError('Dynamic SV requires the expected number of rollouts for every current question')
    for group in groups.values():
        first = group[0]
        if any((r['question'], r['prompt_token_ids']) != (first['question'], first['prompt_token_ids'])
               for r in group):
            raise ValueError('Dynamic SV rollouts for one question have different prompts')
    return list(groups.values())


def generate_references(engine, codec, groups, sampling, *, max_model_len):
    """Generate exactly one unsteered teacher answer per distinct question."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    prompts = [group[0]['prompt_token_ids'] for group in groups]
    if any(not ids or len(ids) + sampling['max_tokens'] > max_model_len for ids in prompts):
        raise ValueError('Dynamic SV teacher reference exceeds the teacher context')
    allowed = codec.valid_ids if codec.valid_ids != list(range(codec.support_size)) else None
    params = SamplingParams(**sampling, stop_token_ids=sorted(codec.eos),
                            output_kind=RequestOutputKind.FINAL_ONLY, allowed_token_ids=allowed)
    outputs = engine.generate([{'prompt_token_ids': ids} for ids in prompts], params,
                              steering=False, use_tqdm=True)
    references = []
    for ids, output in zip(prompts, outputs, strict=True):
        if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
            raise ValueError('Dynamic SV reference generation changed the prompt or sample count')
        references.append(output.outputs[0].text)
    return references


def capture_answer_means(engine, tokenizer, student_tokenizer, groups, references, *,
                         layer, hidden_size, max_context_len):
    """Read both answer sets with the static SV answer-body pooling protocol."""
    from easysteer.capture import capture
    from vllm.capture import SelectSpec

    work = []
    paired_means = []
    for group_index, (group, reference) in enumerate(zip(groups, references, strict=True)):
        prompt = prompt_text(tokenizer, group[0]['question'], enable_thinking=False)
        student_texts = student_tokenizer.batch_decode([row['source_token_ids'] for row in group],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encodings = [encode_answer(tokenizer, prompt, answer, max_context_len)
                     for answer in [*student_texts, reference]]
        usable = all(item['response_ids'] for item in encodings)
        paired_means.append([] if usable else None)
        if usable:
            for item in encodings:
                tokens = item['input_ids']
                positions = list(range(len(item['prompt_ids']), len(tokens)))
                work.append((group_index, tokens, positions))
    if not work:
        return paired_means

    scheduler = engine.llm_engine.vllm_config.scheduler_config
    budgets = capture_budgets(sum(len(positions) for _, _, positions in work),
                              hidden_size, 1, scheduler.max_num_batched_tokens)
    result = capture(engine, [{'prompt_token_ids': tokens} for _, tokens, _ in work],
                     max_tokens=1, layers=[layer], dtype='bfloat16', steering=False, seed=42,
                     per_prompt_selects=[SelectSpec(prompt_positions=positions) for _, _, positions in work],
                     **budgets)
    if list(result.layer_ids) != [layer] or len(result.outputs) != len(work):
        raise ValueError('Dynamic SV capture returned the wrong layer or request count')
    for i, (group_index, tokens, positions) in enumerate(work):
        if (list(result.outputs[i].prompt_token_ids) != tokens or result.sample_positions(i) != positions
                or result.sample_token_ids(i) != [tokens[p] for p in positions]):
            raise ValueError('Dynamic SV capture returned a different prefix or token position')
        states = result.sample(i)[layer]
        if states.shape != (len(positions), hidden_size) or not torch.isfinite(states).all():
            raise ValueError('Dynamic SV capture returned invalid teacher activations')
        paired_means[group_index].append(states.detach().cpu().float().mean(0))
    return paired_means


def direction_from_batch(paired_means, *, layer, operator, alpha):
    differences = [torch.stack(means[:-1]).mean(0) - means[-1]
                   for means in paired_means if means is not None]
    summary = {'usable_questions': len(differences), 'raw_direction_norm': 0., 'injected': False}
    if not differences:
        return {}, summary
    direction = torch.stack(differences).mean(0)
    norm = direction.norm().item()
    if not math.isfinite(norm):
        raise ValueError('Nonfinite dynamic SV direction')
    summary.update(raw_direction_norm=norm, injected=bool(norm >= 1e-8 and alpha))
    if not summary['injected']:
        return {}, summary
    return {layer: direction / norm if operator == 'slerp' else direction * alpha}, summary
