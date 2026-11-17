from common import *
import torch
from directions import build_directions,unit

def test_direction_shapes_zero_leakage_and_multilayer_budget():
 torch.manual_seed(42);n=80;d=8
 b=torch.randn(n,d);a=b+torch.tensor([.5,0.,.3,0.,0.,0.,0.,0.])+torch.randn(n,d)*.1
 specs,vectors,geo=build_directions([str(i) for i in range(n)],a,b,torch.ones(d))
 assert set(vectors)==set(specs)
 assert not specs['negative_direction_control']['eligible'] and not specs['reference_replay']['eligible']
 assert geo['knn_self_excluded'] and geo['knn_neighbors']==64
 for bank in vectors.values():
  for layers in bank.values():
   for value in layers.values():torch.testing.assert_close(value.norm(),torch.tensor(1.))
 torch.testing.assert_close(vectors['reference_replay']['global'][29],unit((a-b).mean(0)))
 torch.testing.assert_close(vectors['negative_direction_control']['global'][29],-vectors['reference_replay']['global'][29])
 assert abs(specs['multilayer_19_29']['alpha']**2*2-.1**2)<1e-12
 assert specs['official_answer_input_mask']['position_mode']=='answer_input'

def test_pilot_only_from_select_and_snapshot_not_shared():
 cfg=read_json(ROOT/'exploration/config.json')
 assert len(cfg['pilot_select_ids'])==64
 assert set(cfg['pilot_select_ids'])<=set(cfg['original_split']['select_ids'])
 assert not set(cfg['pilot_select_ids'])&set(cfg['original_split']['confirm_ids'])
 assert cfg['question_count']==1000 and cfg['gpu_ids']==[0,1]
 from steer_opd import easysteer_static,resumable_core
 assert '/exploration/snapshot/' in easysteer_static.__file__
 assert resumable_core.ROOT==ROOT


def test_adaptive_strength_is_frozen_source_based_and_masks_native_spans():
 from directions import native_strengths
 plan={'units':[{'valid':True,'ranges':{'student':[0,1],'teacher':[0,2]}},
   {'valid':True,'ranges':{'student':[1,2],'teacher':[2,3]}},
   {'valid':True,'ranges':{'student':[2,3],'teacher':[3,4]}}]}
 value=native_strengths({'token_ids':[1,2,3,4,5]},plan,torch.tensor([9/16,12/16,13/16]),'source_adaptive_protect',None)
 torch.testing.assert_close(value,torch.tensor([.3,.3,.2,0.,0.]))


def test_multivector_support_is_declared_only_in_exploration_copy():
 from steer_opd import easysteer_static
 import inspect
 assert "options['steer_multi_vector']=True" in inspect.getsource(easysteer_static.create_engine)
 assert "steer_multi_vector" not in (ROOT/'src/steer_opd/easysteer_static.py').read_text()
