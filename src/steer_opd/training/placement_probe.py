"""Bounded GPU placement probes; ephemeral updates only, no training checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback

from steer_opd.resumable_core import ROOT,atomic_json,read_json


class Monitor:
    def __init__(self,folder):self.folder=folder;self.stop=threading.Event();self.peak={};self.thread=None
    def start(self):
        import subprocess,csv
        self.allowed=set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        def loop():
            while not self.stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
                    for index,uuid,used in csv.reader(raw.splitlines()):
                        if index.strip() in self.allowed or uuid.strip() in self.allowed:
                            key=index.strip();self.peak[key]=max(self.peak.get(key,0),int(used.strip()))
                except Exception:pass
                self.stop.wait(.5)
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=2)


def event(folder,phase,**values):
    value={'phase':phase,'time':time.time(),**values}
    atomic_json(folder/'progress.json',value);print(json.dumps(value),flush=True)


def teacher(args,folder):
    import torch
    from transformers import AutoTokenizer
    from steer_opd.training.sleep_engine import create_sleep_engine,FrozenEngineResidency
    from steer_opd.easysteer_static import fixed_prefix_scores,build_spec
    from steer_opd.static_screen import candidate_vectors
    from steer_opd.compass_alignment import ByteLevelCodec,align_response
    from steer_opd.math_data import prompt_ids
    from steer_opd.training.verl_plan import DEFAULT_BANK,PAIR
    source=ROOT/'outputs/step_3_overlap/ablation'/PAIR/'n1000_slerp0.1_layer26_seed42_v1/source_rows.json'
    originals=read_json(source);row=max(originals,key=lambda r:len(r['token_ids']))
    codecs={r:ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/p,local_files_only=True))
            for r,p in {'student':'assets/models/Qwen3-4B','teacher':'assets/metadata/Qwen3.5-35B-A3B'}.items()}
    aligned=align_response(row['token_ids'],codecs['student'],{'teacher':codecs['teacher']})
    native={'id':'probe','key':'probe','prompt_token_ids':prompt_ids(codecs['teacher'].tokenizer,row['question'],False),
            'token_ids':aligned['native_response_ids']['teacher']}
    started=time.monotonic();event(folder,'teacher_initialization',tp=args.tp,memory_fraction=args.memory,max_model_len=args.context)
    engine=create_sleep_engine(ROOT/'assets/models/Qwen3.5-35B-A3B',codecs['teacher'].support_size,
        max_model_len=args.context,tensor_parallel_size=args.tp,memory=args.memory,enforce_eager=args.eager,
        disable_autotune=args.disable_autotune)
    init=time.monotonic()-started;event(folder,'teacher_initialized',seconds=init)
    bank=torch.load(ROOT/DEFAULT_BANK,map_location='cpu',weights_only=True)
    vectors,_=candidate_vectors(bank,{'id':'probe'},{'direction':'S-T','granularity':'global',
        'operator':'slerp','alpha':.1,'layers':[29]})
    cases=[]
    for target in (1024,8192,args.context):
        total=min(target,args.context);count=total-len(native['prompt_token_ids'])-1
        if count<1:continue
        tokens=(native['token_ids']*math.ceil(count/len(native['token_ids'])))[:count]
        test={**native,'token_ids':tokens};samples=[]
        for repeat in range(2):
            event(folder,'teacher_scoring',target_context=total,response_tokens=count,repeat=repeat)
            spec=build_spec(test,[True]*count,vectors,alpha=.1)
            torch.cuda.synchronize();began=time.monotonic()
            record=fixed_prefix_scores(engine,[test],[spec],include_sampled=True)[0]
            torch.cuda.synchronize();seconds=time.monotonic()-began
            if not torch.isfinite(record['sampled_logps']).all():raise ValueError('Nonfinite observed logps')
            samples.append(seconds)
        cases.append({'kind':'synthetic_capacity_repeated_cached_text','context_tokens_including_completion':total,
                      'batch_size':1,'response_tokens':count,'seconds':samples,'warm_tokens_per_second':count/samples[-1]})
    for total in (8192,args.context):
        count=total-len(native['prompt_token_ids'])-1
        tokens=(native['token_ids']*math.ceil(count/len(native['token_ids'])))[:count]
        tests=[{**native,'id':f'probe_{i}','key':f'probe_{i}','token_ids':tokens} for i in range(4)]
        specs=[build_spec(test,[True]*count,vectors,alpha=.1) for test in tests]
        samples=[]
        for repeat in range(2):
            event(folder,'teacher_scoring_batch',target_context=total,batch_size=4,repeat=repeat)
            torch.cuda.synchronize();began=time.monotonic()
            records=fixed_prefix_scores(engine,tests,specs,include_sampled=True)
            torch.cuda.synchronize();samples.append(time.monotonic()-began)
            if not all(torch.isfinite(record['sampled_logps']).all() for record in records):
                raise ValueError('Nonfinite observed batch logps')
        cases.append({'kind':'synthetic_capacity_repeated_cached_text','context_tokens_including_completion':total,
                      'batch_size':4,'response_tokens':count*4,'seconds':samples,'warm_tokens_per_second':count*4/samples[-1]})
    residency=FrozenEngineResidency(engine);switch=[]
    for _ in range(2):
        began=time.monotonic();residency.sleep();torch.cuda.synchronize();sleep=time.monotonic()-began
        event(folder,'teacher_sleeping',seconds=sleep)
        began=time.monotonic();residency.wake();torch.cuda.synchronize();wake=time.monotonic()-began
        switch.append({'sleep_seconds':sleep,'wake_seconds':wake})
    return {'status':'passed','tp':args.tp,'dtype':'bfloat16','head_dtype':'float32','initialization_seconds':init,
        'memory_fraction':args.memory,'max_model_len':args.context,'enforce_eager':args.eager,
        'disable_autotune':args.disable_autotune,'scoring':cases,'switching':switch,
        'formal_training_updates':0,'teacher_generation_calls':0}


def student(args,folder):
    import torch
    import torch.distributed as dist
    from tensordict import TensorDict
    from steer_opd.training.verl_entry import activate_verl,compose_verl
    activate_verl()
    from steer_opd.training.verl_plan import build_verl_plan
    from steer_opd.training.verl_adapter import verl_loss
    from verl.workers.engine import EngineRegistry
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.utils.padding import left_right_2_no_padding
    from verl.utils import tensordict_utils as tu
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(local_rank),WORLD_SIZE=str(world))
    dist.init_process_group('nccl',init_method='env://' if world>1 else 'file://'+str(folder/'dist_init'),rank=rank,world_size=world)
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'));cfg=compose_verl(plan)
    event(folder,'student_initialization',dtype=cfg.actor_rollout_ref.actor.fsdp_config.model_dtype,world_size=world,rank=rank)
    model=omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
    e=omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.fsdp_config)
    from dataclasses import replace
    e=replace(e,strategy='fsdp',use_dynamic_bsz=False,micro_batch_size_per_gpu=1,
              max_token_len_per_gpu=32768,use_remove_padding=True)
    opt=omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.optim)
    opt=replace(opt,total_training_steps=3,override_optimizer_config={'foreach':False} if args.no_foreach else opt.override_optimizer_config)
    checkpoint=omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.checkpoint)
    engine=EngineRegistry.new(model_type='language_model',backend='fsdp',model_config=model,
        engine_config=e,optimizer_config=opt,checkpoint_config=checkpoint)
    started=time.monotonic();engine.initialize();init=time.monotonic()-started
    source=read_json(ROOT/'outputs/step_3_overlap/ablation'/plan['input_data_dir'].split('outputs/step_0_data/ablation/')[1]/'source_rows.json')
    row=max(source,key=lambda r:len(r['token_ids']));prompt=row['prompt_token_ids']
    cases=[]
    for target in (2048,8192,8192):
        count=target-len(prompt);tokens=(row['token_ids']*math.ceil(count/len(row['token_ids'])))[:count]
        full=prompt+tokens;length=len(full)
        data=TensorDict({'input_ids':torch.tensor([full]),'prompts':torch.tensor([prompt]),
            'responses':torch.tensor([tokens]),'attention_mask':torch.ones(1,length,dtype=torch.long),
            'position_ids':torch.arange(length)[None,:],
            'response_mask':torch.ones(1,count,dtype=torch.long),
            'advantages':torch.ones(1,count)*.1},batch_size=[1])
        data=left_right_2_no_padding(data)
        tu.assign_non_tensor(data,temperature=1.,use_remove_padding=True,use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,max_token_len_per_gpu=32768,calculate_entropy=False,compute_loss=True)
        event(folder,'student_training_probe',context_tokens=target,response_tokens=count)
        torch.cuda.synchronize();began=time.monotonic()
        with engine.train_mode():output=engine.train_batch(data,loss_function=verl_loss)
        torch.cuda.synchronize();seconds=time.monotonic()-began
        cases.append({'context_tokens':target,'response_tokens':count,'seconds':seconds,'tokens_per_second':count/seconds,
                      'grad_norm':str(output.get('metrics',{}).get('grad_norm'))})
        event(folder,'student_probe_step_finished',**cases[-1])
    dist.destroy_process_group()
    return {'status':'passed','world_size':world,'rank':rank,'engine':'verl_fsdp','model_dtype':'fp32','optimizer':'AdamW',
        'cpu_offload':True,'gradient_checkpointing':True,'optimizer_foreach':False if args.no_foreach else None,'initialization_seconds':init,'cases':cases,
        'diagnostic_optimizer_steps':len(cases),'formal_training_updates':0,'checkpoint_saved':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',choices=['teacher','student'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--tp',type=int,default=1)
    p.add_argument('--memory',type=float,default=.98);p.add_argument('--context',type=int,default=16384)
    p.add_argument('--eager',action='store_true');p.add_argument('--no-foreach',action='store_true')
    p.add_argument('--disable-autotune',action='store_true')
    args=p.parse_args();folder=Path(args.output).resolve()
    if int(os.environ.get('WORLD_SIZE','1'))>1:folder=folder/('rank_'+os.environ['RANK'])
    folder.mkdir(parents=True,exist_ok=True)
    if not folder.is_relative_to(ROOT):raise ValueError('Diagnostic outputs must stay in project')
    if not os.environ.get('STEER_OPD_GPU_ADMISSION'):raise RuntimeError('Idle GPU admission is required')
    import torch
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    # Mark the admitted device busy immediately so another launcher sees it.
    guard=torch.empty(1,device='cuda');torch.manual_seed(42)
    monitor=Monitor(folder);monitor.start()
    try:result=teacher(args,folder) if args.role=='teacher' else student(args,folder)
    except BaseException as error:
        result={'status':'failed','role':args.role,'error':repr(error),'traceback':traceback.format_exc(),
                'formal_training_updates':0,'checkpoint_saved':False}
        traceback.print_exc()
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        monitor.close()
    result['peak_device_memory_mib']=monitor.peak
    result['torch_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    result['torch_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    atomic_json(folder/'result.json',result);print(json.dumps(result,default=str),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
