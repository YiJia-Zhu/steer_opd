"""SV-training answer-token means, with the same bounded-context text reading."""
import torch


def encode_answer(tokenizer, prompt_text, text, max_context_len=2048):
    """Match sv_training.acts: tokenize prompt+stripped answer, then keep [:limit].

    The separately tokenized prompt length defines the answer boundary, as in
    upstream mean_hidden_all_layers. Every retained answer token is pooled,
    including reasoning/control markers; correctness and finish_reason are unused.
    """
    if type(max_context_len) is not int or max_context_len < 2:
        raise ValueError('SV max_context_len must be an integer >= 2')
    if getattr(tokenizer, 'truncation_side', 'right') != 'right':
        raise ValueError('SV-training extraction requires right context truncation')
    text = (text or '').strip()
    prefix = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    encoded = tokenizer(prompt_text + text, add_special_tokens=False, return_offsets_mapping=True)
    full = encoded['input_ids']; retained = full[:max_context_len]; boundary = len(prefix)
    response = retained[boundary:] if text and boundary > 0 else []
    offsets = [(max(0, a-len(prompt_text)), max(0, b-len(prompt_text)))
               for a,b in encoded['offset_mapping'][boundary:len(retained)]] if response else []
    return {'input_ids':retained, 'prompt_ids':retained[:boundary], 'response_ids':response,
            'offsets':offsets, 'body_mask':[True]*len(response),
            'original_response_tokens':max(0,len(full)-boundary),
            'retained_response_tokens':len(response), 'max_context_len':max_context_len,
            'context_truncated':len(full)>max_context_len,
            'pooling':'sv_training_all_retained_answer_tokens'}


def pool_question(model, tokenizer, item, records, layers, allow_zero=False, max_context_len=2048):
    captured = {}
    handles = []
    for layer in layers:
        def capture(module, args, output, index=layer):
            captured[index] = output[0] if isinstance(output, tuple) else output
        handles.append(model.model.layers[layer].register_forward_hook(capture))
    result = {str(layer): {"pools": [], "norms": []} for layer in layers}
    device = model.lm_head.weight.device
    try:
        for row in records:
            encoded = encode_answer(tokenizer,item.get('reader_prompt_text',item.get('student_prompt_text')),row.get('text'),max_context_len)
            response, mask = encoded['response_ids'],encoded['body_mask']
            if not response:
                return {"usable": False, "reason": "no_answer_content"}
            prefix = encoded['prompt_ids']
            tokens = torch.tensor([encoded['input_ids']], device=device)
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=tokens, use_cache=False)
            for layer in layers:
                states = captured[layer][0, len(prefix):].float()[torch.tensor(mask, device=device)]
                result[str(layer)]["pools"].append(states.mean(0).cpu())
                result[str(layer)]["norms"].append(states.norm(dim=-1).cpu())
        if len(records) == 2:
            for record in result.values():
                difference = record["pools"][0] - record["pools"][1]
                if not torch.isfinite(difference).all() or (difference.norm() < 1e-8 and not allow_zero):
                    return {"usable": False, "reason": "degenerate_difference_at_candidate_layer"}
        return {"usable": True, "layers": result}
    finally:
        for handle in handles:
            handle.remove()
