"""Launch configuration, GPU isolation and acceptance failures, all on CPU."""
import importlib
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from steer_opd.compass_campaign import make_base_job
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.resumable_core import ROOT,atomic_json,read_json


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('launch_static_panel')


def test_selected_schemes_and_hyperparameters_reach_actual_candidate_jobs(launcher):
    args=launcher.parser().parse_args(['--schemes','2,8','--alpha','.3','--layer','17'])
    cfg=launcher.build_config(args)
    job=make_base_job(cfg,{'student':NS(support_size=151669)},'rows.json')
    assert [c['id'] for c in job['conditions']]==['scheme_02','scheme_08']
    assert [(c['direction'],c['gate'],c['granularity']) for c in job['conditions']]==[
        ('T-S','all','per_question'),('T-R','static_overlap','per_question')]
    assert all(c['alpha']==.3 and c['layers']==[17] for c in job['conditions'])
    assert cfg['confirm_top_n']==2 and len(job['banks'])==4
    assert cfg['optimizer_updates']==0 and cfg['seed']==42
    single=launcher.build_config(launcher.parser().parse_args(['--schemes','scheme_04']))
    assert single['confirm_top_n']==1


@pytest.mark.parametrize('options',[
    ['--schemes','1,1'],['--schemes','9'],['--schemes','typo'],['--alpha','1.1'],['--layer','36'],
    ['--gpu-ids','0','--teacher-tp','2'],['--gpu-ids','0,0'],['--gpu-memory','1'],['--wait-seconds','-1'],
    ['--run-name','../escape'],['--confirm-top-n','0']])
def test_invalid_launch_settings_fail_before_any_gpu_action(launcher,options):
    with pytest.raises(ValueError):launcher.build_config(launcher.parser().parse_args(options))


def test_preview_and_missing_weights_never_launch_or_create_run(launcher,tmp_path,monkeypatch):
    cfg=launcher.build_config(launcher.parser().parse_args([]))
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'build_config',lambda _:cfg)
    monkeypatch.setattr(launcher,'missing_assets',lambda _:[{'role':'teacher','reason':'missing'}])
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('CPU preview launched a subprocess'))
    monkeypatch.setattr(sys,'argv',['launch','--check']);launcher.main()
    monkeypatch.setattr(sys,'argv',['launch','--run'])
    with pytest.raises(FileNotFoundError,match='Weights incomplete'):launcher.main()
    assert not list(tmp_path.iterdir())


def test_changed_run_request_rejected_before_gpu_validation(launcher,tmp_path,monkeypatch):
    cfg=launcher.build_config(launcher.parser().parse_args([]))
    atomic_json(tmp_path/cfg['run_dir']/'launch_request.json',{**cfg,'alpha':.2})
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'build_config',lambda _:cfg)
    monkeypatch.setattr(launcher,'missing_assets',lambda _:[])
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('Changed run reached GPU'))
    monkeypatch.setattr(sys,'argv',['launch','--run'])
    with pytest.raises(ValueError,match='different parameters'):launcher.main()


def test_failed_validation_never_starts_panel(launcher,tmp_path,monkeypatch):
    cfg=launcher.build_config(launcher.parser().parse_args([]))
    shell=tmp_path/'scripts/run_static_panel.sh';shell.parent.mkdir();shell.write_text('# test launch')
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'build_config',lambda _:cfg)
    monkeypatch.setattr(launcher,'missing_assets',lambda _:[])
    commands=[];monkeypatch.setattr(launcher,'run_command',lambda command,env:commands.append(command))
    def failed(*args):raise ValueError('acceptance failed')
    monkeypatch.setattr(launcher,'run_validation',failed)
    monkeypatch.setattr(sys,'argv',['launch','--run'])
    with pytest.raises(ValueError,match='acceptance failed'):launcher.main()
    assert len(commands)==1 and commands[0][-1]=='scripts/check_easysteer_runtime.py'
    assert not (tmp_path/cfg['run_dir']/'launch_config.json').exists()


def test_launcher_subprocess_output_is_logged_and_live(launcher,tmp_path):
    import os
    from steer_opd.experiment_runtime import console_log
    with console_log(tmp_path/'startup.log'):
        launcher.run_command([sys.executable,'-c',"import sys;print('native progress',flush=True);print('stderr',file=sys.stderr)"],os.environ.copy())
    assert 'native progress' in (tmp_path/'startup.log').read_text()
    assert 'stderr' in (tmp_path/'startup.log').read_text()


def test_validation_receipt_required_by_every_generated_stage(tmp_path):
    cfg=read_json(ROOT/'configs/static_dual_overlap.json')
    cfg.update(execution_hold=None,execution_blockers=[],gpu_acceptance_receipt=str(tmp_path/'passed.json'),gpu_acceptance_binding='correct')
    job=make_base_job(cfg,{'student':NS(support_size=151669)},'rows.json')
    with pytest.raises(RuntimeError,match='GPU validation'):require_execution_allowed(job)
    atomic_json(tmp_path/'passed.json',{'status':'passed','binding':'wrong'})
    with pytest.raises(RuntimeError,match='does not match'):require_execution_allowed(job)
    atomic_json(tmp_path/'passed.json',{'status':'passed','binding':'correct'})
    require_execution_allowed(job)


def test_gpu_pool_never_uses_other_free_cards(launcher):
    admission=importlib.import_module('run_on_idle_gpus')
    rows=[{'index':i,'uuid':str(i),'has_compute_process':i==3,'memory_mib':0,'utilization':0} for i in range(5)]
    assert [r['index'] for r in admission.select_idle(rows,rows,set(),1,[3,2])]==[2]
    assert admission.select_idle(rows,rows,{'2'},1,[3,2])==[]
    assert [r['index'] for r in admission.select_idle(rows,rows,set(),2,[4,1])]==[4,1]
    with pytest.raises(ValueError,match='Unknown physical'):admission.select_idle(rows,rows,set(),1,[9])


def test_gpu_probability_validation_rejects_real_mismatches_and_nan(launcher):
    validation=importlib.import_module('validate_static_gpu')
    probabilities=torch.zeros(2,258);probabilities[:,:2]=torch.tensor([[.4,.6],[.5,.5]])
    scores=[{'sampled_logps':torch.tensor([-1.,-2.]),'byte_probs':probabilities}]
    assert validation.compare_scores(scores,scores)['max_abs_logp']==0
    changed=[{**scores[0],'sampled_logps':torch.tensor([-1.,-3.])}]
    with pytest.raises(ValueError,match='parity failed'):validation.compare_scores(scores,changed)
    changed=[{**scores[0],'sampled_logps':torch.tensor([float('nan'),-2.])}]
    with pytest.raises(ValueError,match='parity failed'):validation.compare_scores(scores,changed)


def test_hf_capture_uses_block_output_and_correct_prompt_last_body_rows(launcher):
    from test_static_overlap import tiny,rows
    validation=importlib.import_module('validate_static_gpu')
    model=tiny();model._input_device=torch.device('cpu');batch=rows()
    captured=validation.hf_capture(model,batch,[2])
    with torch.inference_mode():
        for record,states in zip(batch,captured):
            tokens=record['prompt_token_ids']+record['token_ids']
            # Independent block hook verifies the final block output before norm.
            actual=[]
            hook=model.model.layers[2].register_forward_hook(lambda m,a,v:actual.append(v.detach().clone()))
            model.model(input_ids=torch.tensor([tokens]));hook.remove()
            expected=actual[0][0,len(record['prompt_token_ids'])-1:]
            torch.testing.assert_close(states['2'],expected,atol=1e-6,rtol=1e-5)
            assert len(states['2'])==len(record['token_ids'])+1
