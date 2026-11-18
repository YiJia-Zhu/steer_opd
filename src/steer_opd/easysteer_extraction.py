"""Whole-call S0 teacher/self capture with raw answer-body pooling."""


import torch

from steer_opd.compass_job import model_path, project_path
from steer_opd.easysteer_static import create_engine
from steer_opd.extraction import encode_answer
from steer_opd.resumable_core import atomic_json
from steer_opd.experiment_runtime import Progress
from steer_opd.generation_policy import presence_penalty


def capture_budgets(selected_rows, hidden_size, layer_count, scheduled_tokens):
    """Bound host retention separately from per-forward device workspace.

    The pinned capture session estimates graph outputs, selection/cast workspace
    and labels as 3 * tokens * width * 8 + tokens * layers * 12 bytes.
    Use its bound with headroom, without changing the engine's scheduler.
    """
    if any(type(v) is not int or v < 1 for v in (selected_rows, hidden_size, layer_count, scheduled_tokens)):
        raise ValueError('Capture budgeting requires positive row/model/scheduler dimensions')
    page = 16 * 1024**2
    floor = 256 * 1024**2
    retained = selected_rows * layer_count * (hidden_size * 2 + 12)
    workspace = scheduled_tokens * layer_count * (hidden_size * 8 * 3 + 12)
    return {'budget_bytes': max(floor, retained * 2 + page),
            'device_budget_bytes': max(floor, (workspace * 5 + 3)//4 + page)}


def pooled_capture(tokenizer, question, records, captured, layers, max_context_len=2048):
    """Reduce captured answer states in memory; retain only means and norms."""
    result = {str(layer): {"pools": [], "norms": []} for layer in layers}
    for index, record in enumerate(records):
        encoded=encode_answer(tokenizer,question.get('reader_prompt_text',question.get('student_prompt_text')),record.get('text'),max_context_len)
        response,mask=encoded['response_ids'],encoded['body_mask']
        if not response:
            return {"usable": False, "reason": "no_answer_content"}
        states = {layer: value.detach().cpu() for layer, value in captured[index].items()}
        if set(states) != set(layers) or any(v.shape[0] != len(response)+1 or not torch.isfinite(v).all() for v in states.values()):
            raise ValueError("Captured rows do not match retained prompt-last/answer states")
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
    role=config.get('reader_role','student')
    work=[]
    for question,records,identity in inputs:
        encoded=[encode_answer(tokenizer,question.get('reader_prompt_text',question.get('student_prompt_text')),r.get('text'),config['sv_max_context_len']) for r in records]
        if not all(e['response_ids'] for e in encoded):
            pooled[question['id']]={'usable':False,'reason':'no_answer_content','identity':identity}
            continue
        work.extend((question['id'],e['input_ids'],list(range(len(e['prompt_ids'])-1,len(e['input_ids'])))) for e in encoded)
    if not work:
        return
    cfg=AutoConfig.from_pretrained(model_path(config[role]),local_files_only=True)
    cfg=getattr(cfg,'text_config',cfg)
    selected_rows=sum(len(w[2]) for w in work)
    with Progress(output/'capture_progress','activation_capture',len(work),unit='role',show_bar=False) as progress:
        engine=create_engine(model_path(config[role]),len(tokenizer.get_vocab()),
            max_model_len=max(len(w[1]) for w in work)+1,
            gpu_memory_utilization=config.get('gpu_memory_utilization',.75),seed=42,progress=progress,
            batch_invariant=role=='student',
            tensor_parallel_size=config.get('teacher_gpus',2) if role=='teacher' else 1)
        scheduled_tokens=engine.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
        budgets=capture_budgets(selected_rows,cfg.hidden_size,len(layers),scheduled_tokens)
        atomic_json(output/'capture_budget.json', {'selected_rows':selected_rows,
            'hidden_size':cfg.hidden_size,'layers':layers,'scheduled_tokens':scheduled_tokens,
            'capture_dtype':'bfloat16','hidden_state_storage':'memory_only','pooled_dtype':'float32',
            'staging_bytes':16*1024**2,'fetch_bytes':16*1024**2,**budgets})
        progress.event('capture_budget',**budgets,selected_rows=selected_rows,scheduled_tokens=scheduled_tokens)
        class WithProgress:
            def __getattr__(self,name):return getattr(engine,name)
            def generate(self,*args,**kwargs):
                kwargs['use_tqdm']=progress.engine_tqdm
                return engine.generate(*args,**kwargs)
        with progress.phase_scope('continuous_capture'):
            capture=hs.capture(WithProgress(),[{'prompt_token_ids':w[1]} for w in work],
                max_tokens=1,layers=layers,dtype='bfloat16',steering=False,seed=42,
                presence_penalty=presence_penalty(config),
                per_prompt_selects=[SelectSpec(prompt_positions=w[2]) for w in work],
                **budgets)
        if list(capture.layer_ids)!=sorted(layers) or len(capture.outputs)!=len(work):
            raise ValueError('Capture layer/sample count changed')
        progress.finished_call(len(work))
        by_question={}
        for i,(qid,tokens,positions) in enumerate(work):
            if (list(capture.outputs[i].prompt_token_ids)!=tokens or capture.sample_positions(i)!=positions
                    or capture.sample_token_ids(i)!=[tokens[p] for p in positions]):
                raise ValueError('Capture prompt/position/token labels changed')
            if any(v.ndim!=2 or v.shape[1]!=cfg.hidden_size for v in capture.sample(i).values()):
                raise ValueError('Captured hidden width differs from the reader model')
            by_question.setdefault(qid,[]).append(i)
        with progress.phase_scope('pool_activations_in_memory'):
            for question,records,identity in inputs:
                qid=question['id']
                if qid not in by_question:continue
                pooled[qid]={**pooled_capture(tokenizer,question,records,
                    [capture.sample(i) for i in by_question[qid]],layers,config['sv_max_context_len']),
                    'identity':identity}
