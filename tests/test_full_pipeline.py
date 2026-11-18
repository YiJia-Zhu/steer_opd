"""CPU checks for the full launcher's GPU boundary and verified SV handoff."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

from steer_opd.compass_campaign import scheme_conditions
from steer_opd.resumable_core import ROOT, atomic_json


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('run_full_pipeline')


def arguments(launcher, *extra):
    return launcher.parser().parse_args([
        '--input-data-dir', 'frozen_questions', '--base-question-source', 'frozen_base.json', *extra])


@pytest.mark.parametrize('operator,alpha,layer', [('slerp', '.1', 29), ('additive', '.3', 39)])
def test_same_steering_parameters_reach_preparation_and_training(launcher, operator, alpha, layer):
    args = arguments(launcher, '--operator', operator, '--alpha', alpha, '--teacher-layer', str(layer),
                     '--gpu-ids', '2,3', '--wait-seconds', '17', '--sv-max-context-len', '1024')
    cfg, _ = launcher.preparation(args)
    settings = launcher.training_settings(args, cfg)
    [condition] = scheme_conditions(cfg)
    assert (condition['direction'], condition['steer_role'], condition['granularity'], condition['gate']) == (
        'S-T', 'teacher', 'global', 'all')
    assert (condition['operator'], condition['alpha'], condition['layers']) == (operator, float(alpha), [layer])
    assert (settings.operator, settings.alpha, settings.teacher_layer) == (operator, float(alpha), layer)
    assert settings.gpu_pool == (2, 3) and settings.gpu_wait_seconds == 17
    assert cfg['sv_max_context_len'] == 1024 and cfg['data_policy']=='all_train'
    assert cfg['question_count']==1024 and not {'select_fraction','confirm_top_n'}&cfg.keys()
    assert (settings.train_batch_size,settings.total_steps,settings.evaluation_freq)==(32,64,16)
    assert cfg['base_question_source'] == 'frozen_base.json'
    command = launcher.training_command(args, cfg, settings, Path('/resolved/owner/S-T_global.pt'))
    assert command[command.index('--sv-bank') + 1] == '/resolved/owner/S-T_global.pt'
    assert command[command.index('--wait-seconds') + 1] == '17'
    assert '--overlap-metrics' in command and settings.monitor_evaluation
    assert cfg['param_name'].endswith('_st_global_all_full_v2')
    assert command[command.index('--batch-size')+1]=='32'
    assert command[command.index('--total-steps')+1]=='64'
    assert command[command.index('--evaluation-freq')+1]=='16'


@pytest.mark.parametrize('options', [
    ['--sv-granularity', 'per_question'], ['--injection-mode', 'static_overlap'],
    ['--gpu-ids', '0,1,2'], ['--gpu-ids', '0,0'], ['--wait-seconds', '-1'],
    ['--teacher-layer', '40'], ['--alpha', 'nan'], ['--alpha', '1.1'],
    ['--batch-size','64'],['--total-steps','65'],['--total-steps','0']])
def test_unsupported_settings_fail_before_execution(launcher, options):
    with pytest.raises((ValueError, SystemExit)):
        args = arguments(launcher, *options)
        cfg, _ = launcher.preparation(args)
        launcher.training_settings(args, cfg)


def test_cpu_preview_does_not_write_or_start_subprocesses(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['full', '--check', '--input-data-dir', 'frozen_questions',
                                     '--base-question-source', 'frozen_base.json'])
    monkeypatch.setattr(launcher, 'preview_plan', lambda *a: {'output': str(tmp_path), 'retention_policy': 'minimal_v1'})
    monkeypatch.setattr(launcher, 'missing_assets', lambda *a: [])
    monkeypatch.setattr(launcher, 'run_command', lambda *a: pytest.fail('CPU preview started a process'))
    monkeypatch.setattr(launcher, 'frozen_json', lambda *a: pytest.fail('CPU preview wrote a run'))
    launcher.main()
    assert not list(tmp_path.iterdir())


def test_same_pipeline_request_can_resume_without_rewriting_its_identity(launcher, monkeypatch, tmp_path):
    from steer_opd.training import verl_entry
    monkeypatch.setattr(sys, 'argv', ['full', '--run', '--input-data-dir', 'frozen_questions',
                                     '--base-question-source', 'frozen_base.json'])
    monkeypatch.setattr(launcher, 'preview_plan', lambda *a: {'output': str(tmp_path), 'retention_policy': 'minimal_v1'})
    monkeypatch.setattr(launcher, 'missing_assets', lambda *a: [])
    monkeypatch.setattr(launcher, 'metadata_dir', lambda *a: tmp_path)
    monkeypatch.setattr(launcher, 'project', Path)
    monkeypatch.setattr(verl_entry, 'compose_verl', lambda *a: None)
    monkeypatch.setattr(launcher,'prepare_pool',lambda *a:None)
    calls = []
    monkeypatch.setattr(launcher, 'execute', lambda *a: calls.append(a))
    launcher.main()
    request = next(tmp_path.glob('full_pipeline/*/request.json'))
    original = request.read_bytes()
    launcher.main()
    assert request.read_bytes() == original and len(calls) == 2


def test_failed_preparation_cannot_start_training(launcher, monkeypatch):
    args = arguments(launcher)
    cfg, options = launcher.preparation(args)
    settings = launcher.training_settings(args, cfg)
    calls = []
    def fail(command, env):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(launcher, 'run_command', fail)
    monkeypatch.setattr(launcher, 'committed_bank', lambda *a: pytest.fail('Failed preparation reached training'))
    with pytest.raises(subprocess.CalledProcessError):
        launcher.execute(args, cfg, options, settings)
    assert len(calls) == 1 and 'scripts/launch_ablation.py' in calls[0]


def test_completed_preparation_passes_resolved_bank_to_training(launcher, monkeypatch, tmp_path):
    args = arguments(launcher)
    cfg, options = launcher.preparation(args)
    settings = launcher.training_settings(args, cfg)
    atomic_json(tmp_path / 'launch_config.json', {**cfg, 'gpu_acceptance_binding': 'accepted'})
    bank = tmp_path / 'another_parameter/reader_teacher/banks/S-T_global.pt'
    monkeypatch.setattr(launcher, 'metadata_dir', lambda *a: tmp_path)
    monkeypatch.setattr(launcher, 'project', Path)
    monkeypatch.setattr(launcher, 'committed_bank', lambda *a: bank)
    calls = []
    monkeypatch.setattr(launcher, 'run_command', lambda command, env: calls.append(command))
    launcher.execute(args, cfg, options, settings)
    assert len(calls) == 2
    assert calls[1][calls[1].index('--sv-bank') + 1] == str(bank)


def test_bank_handoff_checks_sha_and_reuses_real_owner(launcher, monkeypatch, tmp_path):
    from steer_opd import artifact_reuse, stage_io
    from steer_opd.artifact_reuse import StageArtifacts, contract
    from steer_opd.direction_bank import file_sha256
    monkeypatch.setattr(artifact_reuse, 'ROOT', tmp_path)
    monkeypatch.setattr(stage_io, 'ROOT', tmp_path)
    cfg = {'output_root': 'outputs', 'run_dir': 'outputs/new', 'layout_version': 1}
    owner = tmp_path / 'outputs/old/reader_teacher/banks'
    owner.mkdir(parents=True)
    bank = owner / 'S-T_global.pt'
    bank.write_bytes(b'committed synthetic bank')
    _, packet = StageArtifacts(cfg).publish('bank_teacher', contract('bank_teacher', {'layer': 29}),
                                           owner, files=[bank])
    meta = tmp_path / 'outputs/new/metadata'
    reference = {'descriptor': packet['descriptor_path'],
                 'descriptor_sha256': file_sha256(tmp_path / packet['descriptor_path']),
                 'artifact_id': packet['artifact_id'], 'owner': packet['owner']}
    atomic_json(meta / 'upstreams/bank_teacher.json', reference)
    atomic_json(meta.parent / 'campaign.json', {'status': 'awaiting_user_review',
                'parameter_updates': 0,'data_policy':'all_train','training_questions':1024,
                'evaluated_ids': ['scheme_05']})
    assert launcher.committed_bank(cfg) == bank
    atomic_json(meta / 'upstreams/bank_teacher.json', {**reference, 'descriptor_sha256': 'changed'})
    with pytest.raises(ValueError, match='descriptor changed'):
        launcher.committed_bank(cfg)
    atomic_json(meta / 'upstreams/bank_teacher.json', reference)
    bank.write_bytes(b'changed bank')
    with pytest.raises(ValueError, match='changed'):
        launcher.committed_bank(cfg)


def test_input_fallback_is_read_only_and_prefers_local(launcher, monkeypatch, tmp_path):
    from steer_opd import stage_io
    monkeypatch.setattr(stage_io, 'ROOT', tmp_path)
    relative = 'step_0_data/frozen/questions_source.json'
    remote = tmp_path / 'output_remote' / relative
    remote.parent.mkdir(parents=True)
    remote.write_text('[]')
    assert launcher.frozen_input('outputs/step_0_data/frozen', 'questions_source.json') == 'output_remote/step_0_data/frozen'
    assert not (tmp_path / 'outputs').exists() and remote.read_text() == '[]'
    local = tmp_path / 'outputs' / relative
    local.parent.mkdir(parents=True)
    local.write_text('[]')
    assert launcher.frozen_input('outputs/step_0_data/frozen', 'questions_source.json') == 'outputs/step_0_data/frozen'


def test_whole_training_pool_has_one_overlap_job_and_no_split(launcher,monkeypatch,tmp_path):
    from types import SimpleNamespace as NS
    import steer_opd.compass_campaign as campaign
    from steer_opd.resumable_core import read_json
    args=arguments(launcher)
    cfg,_=launcher.preparation(args)
    data=tmp_path/'data';run=tmp_path/'overlap'
    cfg.update(data_dir=str(data),question_source=str(data/'questions_source.json'),run_dir=str(run),
               stage_inputs={'alignment':'committed_alignment'},allowed_question_ids=['q0'])
    questions=[{'id':f'q{i}','question':f'Question {i}','answer':'0'} for i in range(1024)]
    atomic_json(data/'questions_source.json',questions)
    monkeypatch.setattr(campaign,'prompt_ids',lambda *a,**kw:[1,2])
    monkeypatch.setattr(campaign,'project_path',Path)
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    codecs={r:NS(tokenizer=object()) for r in ('student','teacher')}
    prepared=campaign.prepare_questions(cfg,codecs)
    assert len(prepared)==1024 and not (data/'split.json').exists()
    assert read_json(data/'train_ids.json')['train_ids']==[q['id'] for q in questions]
    rows=[{'id':q['id'],'key':q['id']+'_0','prompt_token_ids':[1,2],'token_ids':[3]} for q in questions]
    monkeypatch.setattr(campaign,'prepare_plans',lambda *a:(rows,{},None,None))
    job=campaign.prepare_all_training_job(cfg,{'conditions':scheme_conditions(cfg)},codecs)
    assert len(read_json(tmp_path/job['rows']))==1024
    assert job['data_policy']=='all_train' and job['output']=='overlap/train'
    assert not (run/'select_rows.json').exists() and not (run/'confirm_rows.json').exists()


def test_legacy_sv_bank_cannot_be_used_with_a_new_training_pool(monkeypatch,tmp_path):
    import torch
    import steer_opd.training.scoring as scoring
    monkeypatch.setattr(scoring,'ROOT',tmp_path)
    monkeypatch.setattr(scoring,'checkpoint_content_id',lambda p:'teacher-content')
    monkeypatch.setattr(scoring,'codecs_for_plan',lambda p:{})
    bank={'identity':{'checkpoint_content_id':'teacher-content','step':0},
          'method':{'direction':'S-T','granularity':'global','layers':[29]}}
    torch.save(bank,tmp_path/'bank.pt')
    plan={'settings':{'sv_bank':'bank.pt','teacher_layer':29},'student':'S','teacher':'T',
          'data_policy':'all_train','question_source_sha256':'1024-source','training_ids_sha256':'1024-ids',
          'training_loss':{'loss_type':'traditional_opd'}}
    with pytest.raises(ValueError,match='same complete training question pool'):
        scoring.validate_assets(plan)
    bank['identity'].update(data_policy='all_train',question_source_sha256='1024-source',training_ids_sha256='1024-ids')
    torch.save(bank,tmp_path/'bank.pt')
    assert scoring.validate_assets(plan)['sv_bank_sha256']
