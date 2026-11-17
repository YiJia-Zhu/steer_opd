import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'outputs/step_3_overlap/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1'
p=next((base/'select/teacher/scheme_05').glob('*.pt'))
v=torch.load(p,map_location='cpu',weights_only=True)
print('record',p.name,{k:list(x.shape) if torch.is_tensor(x) else type(x).__name__ for k,x in v.items()})
pool=ROOT/'outputs/step_2_sv/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_slerp0.1_S26_T29_seed42_st_only_v1/reader_teacher/activations/pooled.pt'
v=torch.load(pool,map_location='cpu',weights_only=True)
print('pooled keys',v.keys());qid=next(iter(v['pooled']));print('pooled row',qid,{k:type(x).__name__ for k,x in v['pooled'][qid].items()});print('identity',v['pooled'][qid].get('identity'))
print('gpu admission runtime independent path possible')
