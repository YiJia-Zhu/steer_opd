"""Baseline selection, sequential failure handling and frozen runtime isolation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    launcher = scripts / 'baselines.sh'
    shutil.copyfile(ROOT / 'scripts/baselines.sh', launcher)
    (scripts / 'runtime_env.sh').write_text('STEER_OPD_PYTHON="$PWD/record_calls"\n')
    interpreter = tmp_path / 'record_calls'
    interpreter.write_text(f'#!{sys.executable}\n' + """\
import json, os, sys
if sys.argv[1] == 'scripts/prepare_baseline_runtime.py':
    print('src')
else:
    with open('calls', 'a') as stream:
        stream.write(json.dumps(sys.argv[1:]) + '\\n')
    if '--loss' in sys.argv and os.environ.get('FAIL_LOSS') == sys.argv[sys.argv.index('--loss') + 1]:
        sys.exit(37)
""")
    interpreter.chmod(0o755)

    def invoke(*args, **env):
        result = subprocess.run(['bash', str(launcher), *args], cwd=tmp_path.parent,
                                env={'PATH': os.environ['PATH'], 'PYTHONPATH': '', **env},
                                capture_output=True, text=True)
        log = tmp_path / 'calls'
        calls = [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []
        return result.returncode, calls
    return invoke


@pytest.mark.parametrize(('mode', 'methods', 'expected'), [
    ([], None, ['traditional_opd', 'compass_opd']),
    (['--run'], 'compassopd vanilla', ['compass_opd', 'traditional_opd']),
    (['--prepare'], 'compassopd', ['compass_opd']),
    (['--check'], 'vanilla', ['traditional_opd']),
])
def test_methods_dispatch_in_order_with_separate_outputs(launch, mode, methods, expected):
    env = {'METHODS': methods} if methods is not None else {}
    code, calls = launch(*mode, **env)
    assert code == 0
    assert [args[args.index('--loss') + 1] for args in calls] == expected
    assert len({args[args.index('--run-name') + 1] for args in calls}) == len(expected)
    for args, loss in zip(calls, expected, strict=True):
        assert args[:3] == ['-m', 'steer_opd.training.verl_entry', *(mode or ['--check'])]
        assert args[args.index('--alpha') + 1] == '0'
        assert '--sv-bank' not in args and '--resume-from' not in args
        assert ('--reference-model' in args) == (loss == 'compass_opd')
        assert ('--anchor-coef' in args) == (loss == 'compass_opd')


@pytest.mark.parametrize('env', [
    {'METHODS': ''}, {'METHODS': 'vanilla typo'}, {'METHODS': 'vanilla vanilla'},
    {'VANILLA_RUN_NAME': 'same', 'COMPASSOPD_RUN_NAME': 'same'},
])
def test_invalid_list_is_rejected_before_first_method(launch, env):
    assert launch('--run', **env) == (2, [])


def test_failed_method_stops_the_suite(launch):
    code, calls = launch('--run', METHODS='compassopd vanilla', FAIL_LOSS='compass_opd')
    assert code == 37 and len(calls) == 1


@pytest.mark.parametrize('sft_steps', ['', '32'])
def test_sft_dispatch_uses_its_own_runner_and_source(launch, sft_steps):
    code, (sft, compass) = launch('--prepare', METHODS='sft compassopd', SFT_TEACHER_DIR='a teacher directory',
                                 SFT_REFERENCE_ANSWERS='a reference file',
                                 TOTAL_STEPS='64', SFT_TOTAL_STEPS=sft_steps, GPU_IDS='4,5')
    assert code == 0
    assert sft[:3] == ['-m', 'steer_opd.sft_baseline', '--prepare']
    assert sft[sft.index('--teacher-dir') + 1] == 'a teacher directory'
    assert sft[sft.index('--reference-answers') + 1] == 'a reference file'
    assert sft[sft.index('--gpu-ids') + 1] == '4,5'
    assert '--loss' not in sft and '--reference-model' not in sft
    if sft_steps:
        assert sft[sft.index('--total-steps') + 1] == sft_steps
    else:
        assert '--total-steps' not in sft
    assert compass[compass.index('--total-steps') + 1] == '64'


def test_sft_expansion_keeps_the_opd_input_separate(launch):
    code, (sft, vanilla) = launch('--check', METHODS='sft vanilla', SFT_DATA_DIR='expanded-sft-data')
    assert code == 0
    assert sft[sft.index('--data-dir') + 1] == 'expanded-sft-data'
    assert sft[sft.index('--run-name') + 1] == 'sft_n4096_correct_seed42_v1'
    assert '--data-dir' not in vanilla


def test_sft_opd_uses_vanilla_with_a_model_only_checkpoint(launch):
    code, (args,) = launch('--check', METHODS='sft_opd', TOTAL_STEPS='32',
                           SFT_OPD_RUN_NAME='opd_after_sft', SFT_OPD_CHECKPOINT='a checkpoint directory')
    assert code == 0
    assert args[:3] == ['-m', 'steer_opd.training.verl_entry', '--check']
    assert args[args.index('--loss') + 1] == 'traditional_opd'
    assert args[args.index('--sft-checkpoint') + 1] == 'a checkpoint directory'
    assert args[args.index('--total-steps') + 1] == '32'
    assert args[args.index('--run-name') + 1] == 'opd_after_sft'
    assert not {'--resume-from', '--sv-bank', '--reference-model'} & set(args)


@pytest.fixture(scope='module')
def baseline_src():
    return Path(subprocess.check_output([sys.executable, str(ROOT / 'scripts/prepare_baseline_runtime.py')],
                                        text=True).strip())


def baseline_module(baseline_src, name):
    spec = importlib.util.spec_from_file_location(name, baseline_src / 'steer_opd/training' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_subprocesses_use_isolated_training_sources(baseline_src):
    src = str(baseline_src)
    check = """
from pathlib import Path
import os, sys
from steer_opd.resumable_core import ROOT, digest, read_json
from steer_opd.experiment_runtime import child_environment
from steer_opd.training import scorer_service, verl_entry, verl_plan, verl_trainer
from steer_opd.training.config import TrainingLossConfig
src = Path(sys.argv[1])
assert all(Path(module.__file__).is_relative_to(src) for module in (scorer_service, verl_entry, verl_plan, verl_trainer))
assert ROOT.resolve() == Path.cwd().resolve()
for python in (sys.executable, ROOT / '.venv-easysteer/bin/python'):
    assert child_environment(python)['PYTHONPATH'].split(os.pathsep)[0] == str(src)
verl_entry.activate_verl()
assert os.environ['PYTHONPATH'].split(os.pathsep)[1] == str(src)
for loss in ('compass_opd', 'traditional_opd'):
    plan = verl_plan.build_verl_plan(read_json(ROOT/'configs/ablation.json'),
        verl_plan.VerlSettings(monitor_evaluation=True), TrainingLossConfig(loss_type=loss))
    plan['run_purpose'] = 'formal_training'
    plan['binding'] = digest({k:v for k,v in plan.items() if k != 'binding'})
    verl_entry.validate_formal_plan(plan)
    assert plan['teacher_steering'] is False and plan['settings']['sv_bank'] is None
    assert str(Path(verl_entry.__file__).relative_to(ROOT)) in plan['implementation_sha256']
"""
    subprocess.run([sys.executable, '-c', check, src], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONPATH': src, 'CUDA_VISIBLE_DEVICES': ''})


def test_completed_method_requires_its_final_checkpoint_and_both_accuracies(tmp_path, baseline_src):
    from steer_opd.resumable_core import atomic_json
    from steer_opd.training.retention import commit_checkpoint
    from test_minimal_retention import write_checkpoint
    entry = baseline_module(baseline_src, 'verl_entry')
    plan = {'output': str(tmp_path), 'binding': 'test', 'retention_policy': 'minimal_v1',
            'settings': {'total_steps': 2, 'gpu_pool': [0, 1]},
            'evaluation': {'counts': {'MATH-500': 500, 'AIME2025': 30}}}
    atomic_json(tmp_path / 'metadata/plan.json', plan)
    checkpoints = tmp_path / 'checkpoints'
    final = write_checkpoint(checkpoints, 2)
    commit_checkpoint(checkpoints, 2, plan)
    assert not entry.completed_baseline(plan)
    metrics = tmp_path / 'evaluation/metrics/step_2.json'
    atomic_json(metrics, {'eval/MATH-500/acc': .5})
    assert not entry.completed_baseline(plan)
    atomic_json(metrics, {'eval/MATH-500/acc': .5, 'eval/AIME2025/acc': 0})
    assert entry.completed_baseline(plan)
    (final / 'data.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint identity'):
        entry.completed_baseline(plan)


def test_compass_reference_tokenization_does_not_change_pairwise_overlap(baseline_src):
    import torch
    from steer_opd.compass_alignment import align_response
    from steer_opd.training.config import TrainingLossConfig
    from steer_opd.training.losses import distillation_loss, training_alignment
    from test_compass_alignment import ToyCodec
    from test_training_overlap import record
    monitor = baseline_module(baseline_src, 'overlap_monitor')
    vocab = {0: b'<eos>', **{i: bytes([96+i]) for i in range(1, 25)}}
    codec = ToyCodec(vocab)
    reference = ToyCodec({**vocab, 25: b'ab'})
    row = {'id': 'r', 'question_id': 'q', 'token_ids': [1, 2, 0]}
    codecs = {'student': codec, 'teacher': codec}
    pair = align_response(row['token_ids'], codec, {'teacher': codec})
    compass = align_response(row['token_ids'], codec, {'teacher': codec, 'reference': reference})
    assert len(pair['units']) == 2 and len(compass['units']) == 1
    student, teacher = record(3), record(2)
    teacher['ids'][1] += 8
    expected = monitor.row_statistics(row, pair, student, teacher, codecs)
    assert monitor.row_statistics(row, compass, student, teacher, codecs) == expected

    settings = TrainingLossConfig(loss_type='compass_opd')
    with_eos = training_alignment(row['token_ids'], codec, codec, settings=settings,
                                  reference_codec=reference, sampled_stop_id=0)
    assert with_eos['units'][-1]['ranges'] == {'student': [2, 3], 'teacher': [2, 3], 'reference': [1, 2]}
    result = distillation_loss(with_eos, torch.tensor([-3., -3., -2.]), torch.tensor([-1., -1., -.5]),
        settings=settings, reference_logps=torch.tensor([-1.5, -1.]), student_anchor_logps=torch.tensor([-3., -3., -2.5]))
    assert result.advantages[-1].item() == pytest.approx(.25)
    teacher_with_eos = {key: torch.cat([value, value[:1]]) for key, value in teacher.items()}
    assert monitor.row_statistics(row, with_eos, student, teacher_with_eos, codecs) == expected
