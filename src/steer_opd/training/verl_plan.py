"""Frozen, CPU-previewable two-GPU shared VERL recipe; does not alter overlap config."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from steer_opd.resumable_core import ROOT, digest, read_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.rollout_seeds import SEED_POLICY
from steer_opd.training.rollout_logprobs import SOURCE as STUDENT_LOGPROB_SOURCE
from steer_opd.training.scoring_request import SCHEMA as SCORING_REQUEST_SCHEMA
from steer_opd.retention import MINIMAL,minimal
from steer_opd.training.paths import training_output_dir
from steer_opd.generation_policy import (DEFAULT_MAX_GENERATION_LENGTH, DEFAULT_PRESENCE_PENALTY, presence_penalty,
    training_sampling_policy, validate_training_sampling_policy)

VERL_REVISION = '61adb0c7eb90730306e7794c2bf248331211e657'
VERL_ARCHIVE_SHA256 = '8fdbb0f301a769b432db0c8fa847ebaa41f0471a91a748b4b2fcd1685ff755fb'
VERL_SOURCE = '.runtime/verl_training/verl-' + VERL_REVISION
PAIR = 'T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/'
DEFAULT_DATA = 'outputs/step_0_data/ablation/' + PAIR + 'n1024_all_train_seed42_v1'
DEFAULT_BANK = ('outputs/step_2_sv/ablation/' + PAIR +
    'n1024_slerp0.1_S26_T29_seed42_trainall_Tgen16384_Sgen16384_Ttemp1_Tp1_pp2_v1_st_global_all_full_v2/reader_teacher/banks/S-T_global.pt')


def gpu_ids(value):
    ids = tuple(int(v) for v in value.split(',')) if isinstance(value, str) else tuple(value)
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('GPU groups must contain unique physical nonnegative IDs')
    return ids


@dataclass(frozen=True)
class VerlSettings:
    gpu_pool: tuple = (0, 1)
    train_batch_size: int = 32
    total_steps: int = 64
    learning_rate: float = 1e-6
    micro_batch_size: int = 1
    max_response_length: int = DEFAULT_MAX_GENERATION_LENGTH
    score_max_model_len: int = 24576
    teacher_layer: int = 29
    alpha: float = 0.1
    operator: str = 'slerp'
    sv_bank: str = DEFAULT_BANK
    teacher_memory: float = 0.75
    reference_memory: float = 0.06
    anchor_memory: float = 0.18
    rollout_memory: float = 0.8
    save_freq: int = 20
    gpu_wait_seconds: int = 3600
    scorer_timeout: int = 1800
    run_name: str = 'verl_st_global_shared2_pp2_len16384_minimal_v1'
    rollout_n: int = 4
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    monitor_overlap: bool = False
    monitor_evaluation: bool = False
    evaluation_freq: int = 16

    def __post_init__(self):
        from steer_opd.operators import validate_operator
        from steer_opd.stage_io import safe_name
        pool=gpu_ids(self.gpu_pool)
        if len(pool)!=2:raise ValueError('Use exactly two shared physical GPUs; teacher remains TP=2')
        if self.train_batch_size > 1024:raise ValueError('Training batch exceeds the unified 1024-question pool')
        if self.train_batch_size < len(pool) or self.train_batch_size % len(pool):
            raise ValueError('Global training batch must divide evenly across student FSDP ranks')
        if min(self.total_steps, self.micro_batch_size, self.max_response_length, self.save_freq) < 1:
            raise ValueError('Positive steps, lengths, microbatch and checkpoint frequency required')
        if not 0 <= self.teacher_layer < 40:raise ValueError('Invalid teacher block')
        validate_operator(self.operator, self.alpha)
        if not 0 < self.learning_rate < 1:raise ValueError('Invalid learning rate')
        if self.score_max_model_len <= 1024 + self.max_response_length:
            raise ValueError('Leave room for teacher retokenization and one completion token')
        if any(not 0 < v < 1 for v in (self.teacher_memory, self.reference_memory, self.anchor_memory, self.rollout_memory)):
            raise ValueError('Memory fractions must be in (0,1)')
        if self.rollout_n<1:raise ValueError('Positive responses per prompt required')
        if self.evaluation_freq<1:raise ValueError('Positive evaluation frequency required')
        if self.gpu_wait_seconds<0:raise ValueError('GPU wait seconds must be nonnegative')
        safe_name(self.run_name)
        presence_penalty({'presence_penalty': self.presence_penalty})


def build_verl_plan(experiment, settings=None, loss=None):
    settings, loss = settings or VerlSettings(), loss or TrainingLossConfig.from_experiment(experiment)
    base = training_output_dir(experiment,settings.run_name,PAIR)
    source = Path(experiment.get('training_input_data_dir',experiment['input_data_dir']))
    questions = ROOT / source / 'questions_source.json'
    original = read_json(questions)
    ids=[q['id'] for q in original]
    if len(original)!=1024 or len(set(ids))!=1024:
        raise ValueError('Use the complete frozen 1024-question training pool with unique IDs')
    if len(original)%settings.train_batch_size:
        raise ValueError('Training batch must divide the full question pool; never drop tail questions')
    inputs=[questions]
    membership=ROOT/source/'train_ids.json'
    if membership.exists():
        record=read_json(membership)
        if record['train_ids']!=ids or record['source_sha256']!=hashlib.sha256(questions.read_bytes()).hexdigest():
            raise ValueError('Unified training question membership changed')
        inputs.append(membership)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in inputs}
    plan = {'schema': 'steer_verl_training_shared_two_gpu_v7', 'verl_revision': VERL_REVISION,
            'verl_source': VERL_SOURCE, 'output': str(base), 'settings': asdict(settings),
            'training_loss': asdict(loss), 'student': experiment['student'], 'teacher': experiment['teacher'],
            'tokenizer_paths': experiment['tokenizer_paths'], 'seed': 42, 'input_sha256': hashes,
            'input_data_dir': str(source), 'data_policy':'all_train','train_split':'all','train_questions':1024,
            'training_ids_sha256':digest(ids),'question_source_sha256':hashes[str(questions.relative_to(ROOT))],
            'automatic_evaluation': settings.monitor_evaluation, 'steering_role': 'teacher', 'direction': 'S-T',
            'granularity': 'global', 'gate': 'all', 'strength_policy': 'fixed',
            'student_rollout_steering': False, 'student_training_steering': False,
            'request_seed_policy': SEED_POLICY,
            'student_logprob_source': STUDENT_LOGPROB_SOURCE,
            'sampling_policy': training_sampling_policy(asdict(settings)),
            'scoring_request_schema': SCORING_REQUEST_SCHEMA,
            'retention_policy':experiment.get('retention_policy',MINIMAL),
            'resource_mode':'shared_two_gpu',
            'gpu_admission':'single_admission_for_whole_pool',
            'phase_order':['student_rollout','student_rollout_logprob_reuse','frozen_scoring','student_update'],
            'on_policy': {'ppo_epochs': 1, 'optimizer_steps_per_rollout_batch': 1,
                          'rollout_n': settings.rollout_n, 'temperature': 1., 'top_p': 1., 'top_k': -1,
                          'presence_penalty': settings.presence_penalty,
                          'sampling_matches_raw_model': settings.presence_penalty == 0.},
            'execution_hold': experiment.get('execution_hold'), 'training_started': False,
            'gpu_training_acceptance': 'pending'}
    # VERL materializes a validation dataset even for a bounded acceptance run.
    # Gold benchmarks stay outside the training pool; only the monitor enables GPU evaluation.
    from steer_opd.training.benchmarks import benchmark_rows
    _,plan['evaluation']=benchmark_rows(original)
    plan['input_sha256'].update(plan['evaluation']['sources'])
    sources = [*Path(__file__).resolve().parent.glob('*.py'),
               *Path(__file__).resolve().parent.glob('*.yaml'), ROOT/'src/steer_opd/compass_alignment.py',
               ROOT/'src/steer_opd/easysteer_static.py', ROOT/'src/steer_opd/easysteer_worker.py',
               ROOT/'scripts/run_on_idle_gpus.py', ROOT/'src/steer_opd/generation_policy.py',
               ROOT/'src/steer_opd/retention.py']
    from steer_opd.training.rethinking_defaults import provenance
    plan['rethinking_defaults']=provenance()
    plan['implementation_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(sources)}
    plan=json.loads(json.dumps(plan))
    plan['binding'] = digest(plan)
    return plan


def verl_overrides(plan):
    """Use the pinned public config; teacher Top-K distillation stays disabled."""
    validate_training_sampling_policy(plan)
    s, root = plan['settings'], ROOT/plan['output']
    if plan.get('student_logprob_source') != STUDENT_LOGPROB_SOURCE:
        raise ValueError('Prepare a new training plan with the vLLM rollout logprob source')
    if plan.get('scoring_request_schema') != SCORING_REQUEST_SCHEMA:
        raise ValueError('Prepare a new training plan with array-backed scoring requests')
    from steer_opd.training.rethinking_defaults import inherited_overrides
    overrides={
        **inherited_overrides(),
        'trainer.use_v1': False, 'model_engine': 'dp',
        'data.train_files': [str(root/'data/train.parquet')],
        'data.val_files': [str(root/'data/benchmarks.parquet')],
        'data.train_batch_size': s['train_batch_size'], 'data.shuffle': False,
        'data.max_prompt_length': 1024, 'data.max_response_length': s['max_response_length'],
        'data.filter_overlong_prompts': True, 'data.truncation': 'error',
        'data.apply_chat_template_kwargs': {'enable_thinking': False},
        'actor_rollout_ref.model.path': str(ROOT/plan['student']),
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.strategy': 'fsdp',
        'actor_rollout_ref.actor.ppo_mini_batch_size': s['train_batch_size'],
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': s['micro_batch_size'],
        'actor_rollout_ref.actor.ppo_epochs': 1, 'actor_rollout_ref.actor.shuffle': False,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':32768,
        'actor_rollout_ref.actor.optim.lr': s['learning_rate'],
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.batch_invariant': True,
        'actor_rollout_ref.model.use_fused_kernels': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.actor.entropy_coeff': 0., 'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.calculate_entropy': False,
        'actor_rollout_ref.rollout.free_cache_engine':True,
        'actor_rollout_ref.rollout.name': 'vllm', 'actor_rollout_ref.rollout.n': s['rollout_n'],
        'actor_rollout_ref.rollout.max_model_len': 1024 + s['max_response_length'],
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': s['rollout_memory'],
        'actor_rollout_ref.rollout.temperature': 1., 'actor_rollout_ref.rollout.top_p': 1.,
        'actor_rollout_ref.rollout.top_k': -1, 'actor_rollout_ref.rollout.seed': 42,
        'actor_rollout_ref.rollout.calculate_log_probs': True,
        'actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode':'raw_logprobs',
        'actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs':16,
        'actor_rollout_ref.rollout.agent.default_agent_loop': 'steer_opd_single_turn',
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(Path(__file__).resolve().parent/'verl_agent.yaml'),
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': s['micro_batch_size'],
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu':32768,
        'algorithm.adv_estimator': 'token_reward_direct',
        'algorithm.use_kl_in_reward': False,
        # Only use the upstream old-policy bypass. The project's registered
        # sampled OPD loss remains in charge; no PPO/IS/RS/centering is enabled.
        'algorithm.rollout_correction': {'bypass_mode':True,'rollout_is':None,
            'rollout_rs':None,'loss_type':'reinforce','score_centering':False},
        'critic.enable': False, 'reward.reward_model.enable': False, 'distillation.enabled': False,
        'reward.custom_reward_function.path': str(Path(__file__).resolve().parent/'verl_reward.py'),
        'reward.custom_reward_function.name': 'compute_score',
        'trainer.n_gpus_per_node': len(s['gpu_pool']), 'trainer.nnodes': 1,
        'trainer.total_epochs': math.ceil(s['total_steps']/max(1,plan['train_questions']//s['train_batch_size'])), 'trainer.total_training_steps': s['total_steps'],
        'trainer.val_before_train': False, 'trainer.test_freq': -1,
        'trainer.save_freq': s['save_freq'], 'trainer.max_actor_ckpt_to_keep': 1, 'trainer.logger': ['console','tensorboard'],
        'trainer.project_name': 'steer_opd', 'trainer.experiment_name': s['run_name'],
        'trainer.default_local_dir': str(root/'checkpoints'), 'trainer.resume_mode': 'auto',
        'ray_kwargs.ray_init.num_gpus': len(s['gpu_pool']),
        'ray_kwargs.ray_init.address':'local',
        'ray_kwargs.ray_init.namespace':s['run_name'],
        'ray_kwargs.ray_init.num_cpus':32,
        'ray_kwargs.ray_init.include_dashboard':False,
        'ray_kwargs.ray_init.object_store_memory':2*1024**3,
        # Keep Unix socket paths below Linux's 107-byte limit, within the
        # existing shorter project alias rather than an external temp folder.
        'ray_kwargs.ray_init._temp_dir':'/home/zhenning/data/working_tmp/steer_opd/r',
        'ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR':str(root/'logs/tensorboard/train'),
    }
    if s['monitor_evaluation']:
        overrides.update({
            'data.val_files':[str(root/'data/benchmarks.parquet')],
            'data.validation_shuffle':False,
            'trainer.val_before_train':True,'trainer.test_freq':s['evaluation_freq'],
            'actor_rollout_ref.rollout.val_kwargs.n':1,
            'actor_rollout_ref.rollout.val_kwargs.do_sample':True,
            'actor_rollout_ref.rollout.val_kwargs.temperature':1.,
            'actor_rollout_ref.rollout.val_kwargs.top_p':1.,
            'trainer.validation_data_dir':str(root/'evaluation/generations'),
            'trainer.rollout_data_dir':str(root/'rollouts'),
        })
    if s.get('monitor_overlap'):
        overrides.update({
            'actor_rollout_ref.rollout.agent.default_agent_loop':'steer_opd_top16_single_turn',
            'actor_rollout_ref.rollout.agent.agent_loop_config_path':str(Path(__file__).resolve().parent/'overlap_agent.yaml'),
            'actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode':'raw_logprobs',
            'actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs':16,
        })
    if minimal(plan):
        overrides.update({'trainer.max_actor_ckpt_to_keep':None,
            'trainer.rollout_data_dir':None,'trainer.validation_data_dir':None,
            'trainer.log_val_generations':0})
    if plan.get('resume_source') and not (root/'checkpoints/latest_checkpointed_iteration.txt').exists():
        overrides.update({'trainer.resume_mode':'resume_path',
                          'trainer.resume_from_path':str(ROOT/plan['resume_source']['checkpoint'])})
    return overrides
