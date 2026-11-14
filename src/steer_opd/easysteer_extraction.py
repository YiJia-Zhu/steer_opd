"""Batched S0 block-output capture with the existing answer-pooling contract."""


import torch

from steer_opd.compass_job import model_path, project_path
from steer_opd.easysteer_static import create_engine
from steer_opd.extraction import encode_content
from steer_opd.resumable_core import atomic_torch
from steer_opd.experiment_runtime import Progress
from steer_opd.direction_bank import file_sha256


def pooled_capture(tokenizer, question, records, captured, layers, save):
    """One capture row for prompt-last, then every observed answer state."""
    result = {str(layer): {"pools": [], "norms": []} for layer in layers}
    prefix = question["student_prompt_ids"]
    for index, record in enumerate(records):
        response, mask, offsets = encode_content(tokenizer, record["text"], with_offsets=True)
        if not any(mask):
            return {"usable": False, "reason": "no_answer_content"}
        states = {layer: value.detach().cpu() for layer, value in captured[index].items()}
        if set(states) != set(layers) or any(v.shape[0] != len(response)+1 or not torch.isfinite(v).all() for v in states.values()):
            raise ValueError("Captured rows do not match full prompt-last/answer states")
        if save is not None:
            save(index, {"student_prompt_ids": torch.tensor(prefix, dtype=torch.int32),
                "student_response_ids": torch.tensor(response, dtype=torch.int32),
                "response_offsets": torch.tensor(offsets, dtype=torch.int32), "body_mask": torch.tensor(mask),
                "offset_semantics": "student_tokenizer_offset_mapping",
                "state_input_positions": torch.arange(len(prefix)-1, len(prefix)+len(response)),
                "prediction_state_rows": torch.arange(len(response)),
                "response_token_state_rows": torch.arange(1, len(response)+1),
                "states": {str(layer): value.detach().cpu().to(torch.bfloat16).contiguous().clone() for layer, value in states.items()},
                "source_dtypes": {str(layer): str(value.dtype) for layer, value in states.items()},
                "cache_dtype": "torch.bfloat16", "steering": False})
        for layer in layers:
            body = states[layer][1:].float()[torch.tensor(mask)]
            result[str(layer)]["pools"].append(body.mean(0).cpu())
            result[str(layer)]["norms"].append(body.norm(dim=-1).cpu())
    return {"usable": True, "layers": result}


def extract_all(config, tokenizer, inputs, pooled):
    """One public capture call; process/persist activations after it returns."""
    if not inputs:
        return
    import easysteer.capture as hs
    from vllm.capture import SelectSpec
    from transformers import AutoConfig
    layers=config['layers'];output=project_path(config['bank_dir'])
    work=[]
    for question,records,identity in inputs:
        encoded=[encode_content(tokenizer,r['text']) for r in records]
        if not all(any(mask) for _,mask in encoded):
            pooled[question['id']]={'usable':False,'reason':'no_answer_content','identity':identity}
            continue
        prefix=question['student_prompt_ids']
        work.extend((question['id'],prefix+tokens,list(range(len(prefix)-1,len(prefix)+len(tokens)))) for tokens,_ in encoded)
    if not work:
        return
    cfg=AutoConfig.from_pretrained(model_path(config['student']),local_files_only=True)
    raw_bytes=sum(len(w[2]) for w in work)*cfg.hidden_size*2*len(layers)
    with Progress(output/'capture_progress','activation_capture',len(work),unit='role',show_bar=False) as progress:
        engine=create_engine(model_path(config['student']),len(tokenizer.get_vocab()),
            max_model_len=max(len(w[1]) for w in work)+1,
            gpu_memory_utilization=config.get('gpu_memory_utilization',.75),seed=42,progress=progress)
        class WithProgress:
            def __getattr__(self,name):return getattr(engine,name)
            def generate(self,*args,**kwargs):
                kwargs['use_tqdm']=progress.engine_tqdm
                return engine.generate(*args,**kwargs)
        with progress.phase_scope('continuous_capture'):
            capture=hs.capture(WithProgress(),[{'prompt_token_ids':w[1]} for w in work],
                max_tokens=1,layers=layers,dtype='bfloat16',steering=False,seed=42,
                per_prompt_selects=[SelectSpec(prompt_positions=w[2]) for w in work],
                budget_bytes=max(256*1024**2,raw_bytes*2))
        if list(capture.layer_ids)!=sorted(layers) or len(capture.outputs)!=len(work):
            raise ValueError('Capture layer/sample count changed')
        progress.finished_call(len(work))
        by_question={}
        for i,(qid,tokens,positions) in enumerate(work):
            if (list(capture.outputs[i].prompt_token_ids)!=tokens or capture.sample_positions(i)!=positions
                    or capture.sample_token_ids(i)!=[tokens[p] for p in positions]):
                raise ValueError('Capture prompt/position/token labels changed')
            by_question.setdefault(qid,[]).append(i)
        roles=('teacher','reference','student')
        with progress.phase_scope('pool_and_save_activations'):
            for question,records,identity in inputs:
                qid=question['id']
                if qid not in by_question:continue
                artifacts={}
                def save(index,payload):
                    path=output/'token_activations'/qid/f'{roles[index]}.pt'
                    atomic_torch(path,{**payload,'identity':identity,'role':roles[index]})
                    artifacts[roles[index]]={'path':str(path.relative_to(project_path('.'))),'sha256':file_sha256(path)}
                pooled[qid]={**pooled_capture(tokenizer,question,records,
                    [capture.sample(i) for i in by_question[qid]],layers,save),
                    'identity':identity,'token_activations':artifacts}
