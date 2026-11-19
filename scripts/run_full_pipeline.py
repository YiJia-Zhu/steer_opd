#!/usr/bin/env python3
"""Prepare teacher-global SV, verify one all-position candidate, then train OPD."""
from dataclasses import asdict
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys

from launch_ablation import build_config, parser as ablation_parser, run_command
from run_ablation import missing_assets
from steer_opd.artifact_reuse import StageArtifacts
from steer_opd.compass_campaign import frozen_json
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, digest, read_json
from steer_opd.stage_io import component_target, frozen_data_dir, metadata_dir, project
from steer_opd.tensor_checkpoint import DEFAULT_CANDIDATE_FORMAT, SUPPORTED_FORMATS
from steer_opd.retention import ALL_MODELS_LATEST_STATE,CHECKPOINT_POLICIES
from steer_opd.training.verl_plan import DEFAULT_DATA, VerlSettings, build_verl_plan, gpu_ids


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='CPU preview; no run files or GPU work')
    modes.add_argument('--run', action='store_true', help='Start/resume preparation and formal training')
    modes.add_argument('--prepare-data', action='store_true', help='CPU-only: preserve the old 1000 questions and freeze 24 additional questions')
    p.add_argument('--sv-granularity', choices=['global'], default='global',
                   help='OPD currently supports global; per_question is overlap-only')
    p.add_argument('--injection-mode', choices=['all'], default='all',
                   help='OPD currently supports all; static_overlap is overlap-only')
    p.add_argument('--operator', choices=['slerp', 'additive'], default='slerp')
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--teacher-layer', type=int, default=29)
    p.add_argument('--sv-max-context-len', type=int, default=2048)
    p.add_argument('--question-count',type=int,choices=[1024],default=1024)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--total-steps',type=int,default=64)
    p.add_argument('--evaluation-freq',type=int,default=16)
    p.add_argument('--save-freq',type=int,default=16)
    p.add_argument('--checkpoint-retention',choices=CHECKPOINT_POLICIES,default=ALL_MODELS_LATEST_STATE)
    p.add_argument('--gpu-ids', default='0,1')
    p.add_argument('--wait-seconds', type=int, default=3600)
    p.add_argument('--prep-gpu-memory', type=float, default=.75)
    p.add_argument('--presence-penalty', type=float, default=2.)
    p.add_argument('--reference-max-tokens', type=int, default=16384)
    p.add_argument('--source-max-tokens', type=int, default=16384)
    p.add_argument('--overlap-metrics', type=int, choices=[0, 1], default=1)
    p.add_argument('--hf-parity', type=int, choices=[0, 1], default=0)
    p.add_argument('--candidate-tensor-format', choices=SUPPORTED_FORMATS, default=DEFAULT_CANDIDATE_FORMAT)
    p.add_argument('--input-data-dir', default='')
    p.add_argument('--base-question-source', default='')
    p.add_argument('--param-name', default='')
    p.add_argument('--train-run-name', default='')
    p.add_argument('--train-output-root', default='outputs')
    return p


def frozen_input(relative, filename):
    """Use only known small input files from the read-only remote mount if needed."""
    local = Path(relative)
    remote = Path('output_remote') / local.relative_to('outputs')
    for candidate in (local, remote):
        if project(candidate / filename).is_file():
            return str(candidate)
    raise FileNotFoundError(f'Frozen input is missing: {local / filename}; '
                            'mount output_remote or set the input path explicitly')


def preparation(args):
    # One teacher S-T/global/all candidate, with the shared no-SV baseline.
    options = ['--teacher-ablation', '1', '--schemes', '5', '--data-policy','all_train',
               '--stop-after', 'panel', '--strength-policies', 'fixed']
    for name, value in {
        'gpu-ids': args.gpu_ids, 'teacher-tp': 2, 'teacher-layer': args.teacher_layer,
        'operator': args.operator, 'alpha': args.alpha, 'sv-max-context-len': args.sv_max_context_len,
        'wait-seconds': args.wait_seconds, 'gpu-memory': args.prep_gpu_memory,
        'presence-penalty': args.presence_penalty, 'reference-max-tokens': args.reference_max_tokens,
        'source-max-tokens': args.source_max_tokens, 'hf-parity': args.hf_parity,
        'candidate-tensor-format': args.candidate_tensor_format, 'question-count':args.question_count,
        'input-data-dir': args.input_data_dir or DEFAULT_DATA,
        'base-question-source': args.base_question_source or str(
            Path(frozen_input(frozen_data_dir(1000), 'questions_source.json')) / 'questions_source.json'),
    }.items():
        options += ['--' + name, str(value)]
    cfg = build_config(ablation_parser().parse_args(options))
    name = args.param_name or cfg['param_name'] + '_st_global_all_full_v3'
    options += ['--param-name', name]
    return build_config(ablation_parser().parse_args(options)), options


def training_settings(args, cfg):
    # Build from the same options as preparation; never fall back to DEFAULT_BANK.
    if args.batch_size!=32 or not 1<=args.total_steps<=64:
        raise ValueError('The full pipeline uses 32 questions per step and at most 64 steps')
    return VerlSettings(
        gpu_pool=gpu_ids(args.gpu_ids), run_name=args.train_run_name or 'opd_' + cfg['param_name'],
        teacher_layer=args.teacher_layer, operator=args.operator, alpha=args.alpha,
        sv_bank=str(component_target(cfg, 'bank_teacher') / 'S-T_global.pt'),
        presence_penalty=args.presence_penalty, monitor_overlap=bool(args.overlap_metrics),
        monitor_evaluation=True, gpu_wait_seconds=args.wait_seconds,
        train_batch_size=args.batch_size,total_steps=args.total_steps,evaluation_freq=args.evaluation_freq,
        save_freq=args.save_freq,checkpoint_retention=args.checkpoint_retention)


def prepare_pool(cfg):
    """Freeze the unified CPU input pool; leave historical local/remote inputs intact."""
    from prepare_ablation_data import prepare
    folder=project(cfg['input_data_dir'])
    data_cfg={**cfg,'layout_version':0,'data_dir':str(folder),
              'question_source':str(folder/'questions_source.json')}
    prepare(data_cfg)


def preview_plan(args, cfg, settings):
    from steer_opd.training.verl_entry import validate_formal_plan
    plan = build_verl_plan({**cfg, 'train_output_root': args.train_output_root}, settings)
    plan['run_purpose'] = 'formal_training'
    plan['binding'] = digest({k: v for k, v in plan.items() if k != 'binding'})
    validate_formal_plan(plan)
    return plan


def committed_bank(cfg):
    """Follow the frozen upstream reference, including a reused bank's real owner."""
    run = project(cfg['run_dir'])
    campaign = read_json(run / 'campaign.json')
    if (campaign.get('status') != 'awaiting_user_review' or campaign.get('parameter_updates') != 0
            or campaign.get('data_policy')!='all_train' or campaign.get('training_questions')!=1024
            or campaign.get('evaluated_ids') != ['scheme_05']):
        raise ValueError('Teacher-global/all overlap preparation did not finish')
    reference = read_json(project(metadata_dir(cfg)) / 'upstreams/bank_teacher.json')
    descriptor = project(reference['descriptor'])
    if file_sha256(descriptor) != reference['descriptor_sha256']:
        raise ValueError('Frozen teacher bank descriptor changed')
    owner, packet = StageArtifacts(cfg).validate(descriptor)
    if (packet['component'] != 'bank_teacher' or packet['artifact_id'] != reference['artifact_id']
            or owner != project(reference['owner']) or 'S-T_global.pt' not in packet['files']):
        raise ValueError('Frozen teacher bank owner or identity changed')
    return owner / 'S-T_global.pt'


def training_command(args, cfg, settings, bank):
    command = [sys.executable, '-m', 'steer_opd.training.verl_entry', '--run',
               '--config', str(project(metadata_dir(cfg)) / 'launch_config.json'),
               '--loss', 'traditional_opd', '--gpu-ids', args.gpu_ids,
               '--wait-seconds', str(settings.gpu_wait_seconds),
               '--batch-size',str(settings.train_batch_size),'--total-steps',str(settings.total_steps),
               '--evaluation-freq',str(settings.evaluation_freq),
               '--save-freq',str(settings.save_freq),'--checkpoint-retention',settings.checkpoint_retention,
               '--run-name', settings.run_name, '--sv-bank', str(bank),
               '--teacher-layer', str(settings.teacher_layer), '--operator', settings.operator,
               '--alpha', str(settings.alpha), '--presence-penalty', str(settings.presence_penalty),
               '--train-output-root', args.train_output_root]
    if settings.monitor_overlap:
        command.append('--overlap-metrics')
    return command


def execute(args, cfg, options, settings):
    # The existing launcher owns acceptance/admission and preserves the held template.
    run_command([sys.executable, 'scripts/launch_ablation.py', '--run', *options], os.environ.copy())
    saved = read_json(project(metadata_dir(cfg)) / 'launch_config.json')
    acceptance_keys = {'gpu_acceptance_binding', 'gpu_acceptance_receipt'}
    if ({k: v for k, v in saved.items() if k not in acceptance_keys}
            != {k: v for k, v in cfg.items() if k not in acceptance_keys}):
        raise ValueError('Completed preparation config differs from the requested parameters')
    bank = committed_bank(cfg)
    print(f'SV bank 已校验：{bank}\n开始 OPD 训练及裸学生评测。', flush=True)
    run_command(training_command(args, cfg, settings, bank), os.environ.copy())


def main():
    args = parser().parse_args()
    cfg, options = preparation(args)
    settings = training_settings(args, cfg)
    if args.prepare_data or (args.run and not (project(cfg['input_data_dir'])/'questions_source.json').is_file()):
        prepare_pool(cfg)
    if args.prepare_data:
        print(f"1024题训练池已冻结（CPU）：{cfg['input_data_dir']}",flush=True)
        return
    plan = preview_plan(args, cfg, settings)
    print(f"PARAM_NAME={cfg['param_name']}\nTRAIN_RUN_NAME={settings.run_name}", flush=True)
    print(f'教师空间 S-T / global / all / {settings.operator} alpha={settings.alpha:g} / '
          f'层 {settings.teacher_layer} / SV上下文 {args.sv_max_context_len}', flush=True)
    print(f'物理GPU池：{args.gpu_ids}；同一1024题用于SV、overlap及OPD；最多{settings.total_steps}步。', flush=True)
    print(f'训练：{settings.train_batch_size}题×4条/step，lr=1e-6，seed42/43/44/45；固定SV与强度。', flush=True)
    print(f'评测：step0、每{settings.evaluation_freq}步及末步；裸学生 MATH-500(500) + AIME2025(30)，每题1条。', flush=True)
    print(f'模型checkpoint：每{settings.save_freq}步及末步；策略={settings.checkpoint_retention}。',flush=True)
    print(f"准备输出：{cfg['run_dir']}\n训练输出：{plan['output']}\n存储：{plan['retention_policy']}", flush=True)
    print(f"冻结题集（只读）：{cfg['input_data_dir']}", flush=True)
    missing = missing_assets(cfg)
    print(f'模型文件：{missing or "已存在；实际启动时校验内容身份"}', flush=True)
    if not args.run:
        print('CPU预览完成。启动：bash scripts/run_full_pipeline.sh --run', flush=True)
        return
    if missing:
        raise FileNotFoundError('模型文件不完整，请先完成手动准备')
    if not (ROOT / cfg['engine_python']).is_file():
        raise FileNotFoundError('Prepare the isolated EasySteer environment first')
    # Check the pinned trainer config on CPU before any expensive preparation.
    from steer_opd.training.verl_entry import compose_verl
    compose_verl(plan)
    meta = project(metadata_dir(cfg))
    meta.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (meta / 'full_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {'schema': 'teacher_global_all_full_pipeline_v3', 'preparation': cfg,
                   'training_settings': {**asdict(settings), 'gpu_pool': list(settings.gpu_pool)},
                   'train_output_root': args.train_output_root,
                   'launcher_sources': {name: (ROOT / name).read_text() for name in
                                        ('scripts/run_full_pipeline.sh', 'scripts/run_full_pipeline.py')}}
        frozen_json(meta / 'full_pipeline' / settings.run_name / 'request.json', request)
        execute(args, cfg, options, settings)
    print(f"全流程完成。模型checkpoint、最新恢复状态、训练标量及评测：{plan['output']}", flush=True)


if __name__ == '__main__':
    main()
