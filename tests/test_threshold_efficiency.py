"""Strict threshold equivalence and durable reuse of source overlap metrics."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from steer_opd.overlap_tuning import calibrate_threshold
from steer_opd.resumable_core import atomic_json, atomic_torch, read_json


def exhaustive_threshold(values, target, comparison, lower_bound):
    candidates = set(values)
    candidates.update(math.nextafter(v, math.inf if comparison == 'lt' else -math.inf) for v in values)
    if lower_bound is not None:
        candidates = {max(lower_bound, v) for v in candidates}
    def rate(t):
        return sum(v < t if comparison == 'lt' else v > t for v in values)/len(values)
    chosen = min(candidates, key=lambda t:(abs(rate(t)-target),t))
    return {'threshold':chosen,'target_rate':target,'observed_rate':rate(chosen),
            'positions':len(values),'comparison':comparison}


@pytest.mark.parametrize('comparison',['lt','gt'])
@pytest.mark.parametrize('lower_bound',[None,0.,.4])
def test_histogram_matches_exhaustive_strict_boundaries_and_ties(comparison,lower_bound):
    rng=np.random.default_rng(42)
    cases=[[0.,0.,.5,.5,1.], [.5]*7, list(rng.integers(0,17,67)/16),
           rng.uniform(-.2,1.2,47).tolist(), [.5,math.nextafter(.5,math.inf),.5,1.]]
    for values in cases:
        for target in (.01,.2,.5,.67,.99):
            assert calibrate_threshold(iter(values),target,comparison,lower_bound=lower_bound)==exhaustive_threshold(
                values,target,comparison,lower_bound)


def test_continuous_scores_use_one_input_pass_and_preserve_exact_target():
    class Once:
        used=False
        def __iter__(self):
            assert not self.used
            self.used=True
            return ((i+.5)/50000 for i in range(50000))
    result=calibrate_threshold(Once(),.2,'lt',lower_bound=0.)
    assert result['threshold']==math.nextafter(9999.5/50000,math.inf)
    assert result['observed_rate']==.2 and result['positions']==50000
    for values in ([],[float('nan')],[float('inf')]):
        with pytest.raises(ValueError,match='finite'):calibrate_threshold(values,.2,'lt')


def test_source_metric_cache_reuses_results_and_rejects_changed_inputs_or_bytes(tmp_path,monkeypatch):
    from steer_opd import compass_job as module
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    calls=[]
    class Adapter:
        plans={'q':{'alignment_id':'same_prefix'}}
        def __call__(self,key,student,teacher):
            calls.append(key)
            return {'overlap':torch.tensor([.5,float('nan')],dtype=torch.float64),
                'P_overlap':torch.tensor([.4,float('nan')],dtype=torch.float64),
                'alignment_mask':torch.tensor([True,False]),'alignment_id':'same_prefix',
                'metric_version':4,'matching_rule_version':'native_token_prefix_compatibility_v1',
                'P_overlap_definition':'top16_raw_probability_maximum_prefix_compatible_mass',
                'topk_tie_policy':'engine_native_topk_ranks_retained_no_posthoc_boundary_expansion'}
    args=({'source_dir':'source'},{'id':'q','key':'q'},None,None,Adapter(),'student_sha','teacher_sha')
    first=module.cached_source_metrics(*args)
    second=module.cached_source_metrics(*args)
    assert calls==['q']
    torch.testing.assert_close(first['overlap'],second['overlap'],equal_nan=True)
    with pytest.raises(ValueError,match='inputs'):
        module.cached_source_metrics(*args[:-2],'different_student_sha','teacher_sha')
    path=tmp_path/'alignment/q.metrics.pt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):module.cached_source_metrics(*args)
    # A partial write without its checksum marker is recomputed on CPU.
    (tmp_path/'alignment/q.metrics.json').unlink()
    module.cached_source_metrics(*args)
    assert calls==['q','q']


@pytest.mark.parametrize('gate_metric',['overlap','P_overlap'])
def test_preparation_calibrates_both_metrics_and_uses_the_selected_gate(tmp_path,monkeypatch,gate_metric):
    from steer_opd import compass_campaign as module
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    rows=[{'id':'select','key':'select','model':'S','rollout_topk':'topk.pt'},
          {'id':'confirm','key':'confirm','model':'S','rollout_topk':'topk.pt'}]
    job={'student':'S','source_dir':'source','support_size':32,'teacher_scores':'teacher','require_dual_metrics':True}
    native=[{'key':row['key'],'token_ids':[1,2]} for row in rows]
    monkeypatch.setattr(module,'prepare_plans',lambda *a:(rows,{}, {'teacher':native},{}))
    score={'ids':torch.arange(16,dtype=torch.int32).repeat(2,1),
           'logps':torch.full((2,16),-4.),'binding':'original'}
    atomic_torch(tmp_path/'teacher/select.pt',score)
    (tmp_path/'source').mkdir();(tmp_path/'source/topk.pt').write_bytes(b'original')
    atomic_json(tmp_path/'data/split.json',{'select_ids':['select'],'confirm_ids':['confirm']})
    monkeypatch.setattr(module,'native_score_binding',lambda *a:'original')
    monkeypatch.setattr(module,'load_rollout_topk',lambda *a:score)
    monkeypatch.setattr(module,'AlignedMetricAdapter',lambda *a,**kw:None)
    monkeypatch.setattr(module,'cached_source_metrics',lambda *a:{
        'overlap':torch.tensor([0.,.5]),'P_overlap':torch.tensor([.1,.6]),'alignment_mask':torch.tensor([True,True])})
    cfg={'data_dir':'data','run_dir':'run','minimum_split_questions':1,'gate_metric':gate_metric,'gate_target_rate':.2}
    module.prepare_screen_jobs(cfg,job,{'student':SimpleNamespace(support_size=32),
                                     'teacher':SimpleNamespace(support_size=32)})
    calibrations=read_json(tmp_path/'run/thresholds.json')['metrics']
    assert set(calibrations)=={'overlap','P_overlap'}
    chosen=read_json(tmp_path/'run/threshold.json')
    assert chosen['metric']==gate_metric and chosen['threshold']==calibrations[gate_metric]['threshold']
