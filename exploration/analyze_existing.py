"""Read-only evidence on the frozen layer29 alpha .1 result."""
from common import *
import torch
from collections import defaultdict

def analyze():
 result={}
 for split in ('select','confirm'):
  path=ST/split/'teacher'
  aggregate=defaultdict(lambda:[0.,0.,0.,0.,0.]);question_gains=[]
  for p in sorted((path/'scheme_05').glob('*.pt')):
   v=load_packet(p);base=load_packet(path/'baseline'/p.name)
   valid=v['alignment_mask'];delta=v['overlap'].double()-base['overlap'].double()
   question_gains.append(float(delta[valid].mean())*100)
   idx=torch.arange(len(valid))
   bins={'all':valid,'response_idx_lt512':valid&(idx<512),'response_idx_512_2047':valid&(idx>=512)&(idx<2048),
     'response_idx_ge2048':valid&(idx>=2048),'low_baseline_lt0.5625':valid&(base['overlap']<.5625),
     'middle_baseline_0.5625_0.8125':valid&(base['overlap']>=.5625)&(base['overlap']<.8125),
     'high_baseline_ge0.8125':valid&(base['overlap']>=.8125)}
   for name,mask in bins.items():
    d=delta[mask];a=aggregate[name];a[0]+=len(d);a[1]+=float(d.sum());a[2]+=int((d>0).sum());a[3]+=int((d<0).sum());a[4]+=float(base['overlap'][mask].double().sum())
  result[split]={'questions':len(question_gains),'questions_improved':sum(v>0 for v in question_gains),
   'question_mean_delta_pp':sum(question_gains)/len(question_gains),
   'positions':{k:{'n':int(v[0]),'baseline_token_value':v[4]/v[0],'token_delta_pp':100*v[1]/v[0],
     'improved_fraction':v[2]/v[0],'degraded_fraction':v[3]/v[0],'unchanged_fraction':1-(v[2]+v[3])/v[0]} for k,v in aggregate.items()}}
 atomic_json(SUMMARY/'existing_effect_diagnostics.json',result)
 print(result,flush=True)
if __name__=='__main__':analyze()
