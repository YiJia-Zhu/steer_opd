"""Gold benchmark separation and real TensorBoard scalar persistence."""
from dataclasses import replace

import pytest

from steer_opd.resumable_core import ROOT,read_json,digest
from steer_opd.training.benchmarks import benchmark_rows
from steer_opd.training.verl_plan import VerlSettings,build_verl_plan,verl_overrides
from steer_opd.training.verl_entry import activate_verl,validate_formal_plan,prepare_data


def test_monitoring_plan_uses_gold_benchmarks_and_one_sample_accuracy():
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'),VerlSettings(monitor_evaluation=True))
    assert plan['evaluation']['counts']=={'MATH-500':500,'AIME2025':30}
    cfg=verl_overrides(plan)
    assert cfg['trainer.logger']==['console','tensorboard']
    assert cfg['trainer.test_freq']==16 and cfg['trainer.val_before_train']
    assert cfg['actor_rollout_ref.rollout.val_kwargs.n']==1
    assert cfg['data.val_files'][0].endswith('benchmarks.parquet')
    assert cfg['trainer.resume_mode']=='auto'
    plan['run_purpose']='formal_training';plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    validate_formal_plan(plan)
    plan['settings']['train_batch_size']=2
    plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    with pytest.raises(ValueError,match='authorized'):validate_formal_plan(plan)


def test_benchmark_overlap_with_sv_data_cannot_be_hidden():
    rows,_=benchmark_rows([])
    question=rows[0]['extra_info']['question']
    with pytest.raises(ValueError,match='duplicates an SV'):benchmark_rows([{'question':question}])


def test_training_stays_ungraded_while_benchmark_uses_existing_verifier():
    activate_verl()
    from steer_opd.training.verl_reward import compute_score
    assert compute_score('steer_opd','wrong answer','')==0.
    assert compute_score('steer_eval_AIME2025',r'\boxed{70}','70')['acc']==1.
    assert compute_score('steer_eval_AIME2025',r'\boxed{71}','70')['acc']==0.
    assert compute_score('steer_eval_MATH-500',r'\boxed{\frac{1}{2}}',r'\frac{1}{2}')['acc']==1.
    with pytest.raises(ValueError,match='gold answer'):compute_score('steer_eval_MATH-500','answer','')


def test_native_verl_tensorboard_writes_loss_and_true_accuracy(tmp_path,monkeypatch):
    activate_verl()
    monkeypatch.setenv('TENSORBOARD_DIR',str(tmp_path))
    from verl.utils.tracking import Tracking
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tracking=Tracking('steer_opd','test',default_backend=['tensorboard'])
    tracking.log({'actor/loss':.25,'eval/AIME2025/acc':.5},step=20);tracking.finish()
    saved=EventAccumulator(str(tmp_path)).Reload()
    assert saved.Scalars('actor/loss')[0].step==20
    assert saved.Scalars('actor/loss')[0].value==.25
    assert saved.Scalars('eval/AIME2025/acc')[0].value==.5


def test_gold_parquet_is_separate_from_1024_train_questions(tmp_path):
    import pandas as pd
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'),VerlSettings(monitor_evaluation=True))
    plan['output']=str(tmp_path);prepare_data(plan)
    assert len(pd.read_parquet(tmp_path/'data/train.parquet'))==1024
    benchmark=pd.read_parquet(tmp_path/'data/benchmarks.parquet')
    assert len(benchmark)==530 and all(r['ground_truth'] for r in benchmark.reward_model)


def test_numpy_validation_metrics_persist_as_numbers(tmp_path):
    import json
    import numpy as np
    from steer_opd.resumable_core import atomic_json
    from steer_opd.training.verl_trainer import serializable_metrics
    metrics={'val-aux/num_turns/max':np.int32(2),'eval/MATH-500/acc':np.float64(.75),
             'nested':[np.int64(530),{'finite':np.bool_(True)}]}
    path=tmp_path/'step_0.json';atomic_json(path,serializable_metrics(metrics))
    result=json.loads(path.read_text())
    assert result=={'val-aux/num_turns/max':2,'eval/MATH-500/acc':.75,'nested':[530,{'finite':True}]}
    assert type(result['val-aux/num_turns/max']) is int
