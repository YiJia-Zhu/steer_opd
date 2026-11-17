"""Frozen, CPU-previewable four-GPU VERL recipe; does not alter overlap config."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from steer_opd.resumable_core import ROOT, digest, read_json
from steer_opd.training.config import TrainingLossConfig

VERL_REVISION = '61adb0c7eb90730306e7794c2bf248331211e657'
VERL_ARCHIVE_SHA256 = '8fdbb0f301a769b432db0c8fa847ebaa41f0471a91a748b4b2fcd1685ff755fb'
VERL_SOURCE = '.runtime/verl_training/verl-' + VERL_REVISION
PAIR = 'T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/'
DEFAULT_BANK = 'outputs/step_2_sv/ablation/' + PAIR + 'n1000_slerp0.1_S26_T29_seed42_st_only_v1/reader_teacher/banks/S-T_global.pt'


def gpu_ids(value):
    ids = tuple(int(v) for v in value.split(',')) if isinstance(value, str) else tuple(value)
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('GPU groups must contain unique physical nonnegative IDs')
    return ids


@dataclass(frozen=True)
class VerlSettings:
    student_gpus: tuple = (0, 1)
    teacher_gpus: tuple = (2, 3)
    train_batch_size: int = 16
    total_steps: int = 100
    learning_rate: float = 1e-6
    micro_batch_size: int = 1
    max_response_length: int = 7168
    score_max_model_len: int = 16384
    teacher_layer: int = 29
    alpha: float = 0.1
    operator: str = 'slerp'
    sv_bank: str = DEFAULT_BANK
    teacher_memory: float = 0.60
    reference_memory: float = 0.06
    anchor_memory: float = 0.18
    rollout_memory: float = 0.4
    save_freq: int = 10
    gpu_wait_seconds: int = 3600
    scorer_timeout: int = 1800
    run_name: str = 'verl_st_global_fixed_v1'

    def __post_init__(self):
        from steer_opd.operators import validate_operator
        from steer_opd.stage_io import safe_name
        student, teacher = gpu_ids(self.student_gpus), gpu_ids(self.teacher_gpus)
        if set(student) & set(teacher):raise ValueError('Student FSDP and EasySteer teacher GPU groups must be disjoint')
        if len(teacher) != 2:raise ValueError('Current Qwen3.5 teacher recipe uses TP=2')
        if self.train_batch_size > 700:raise ValueError('Training batch exceeds the frozen Select pool')
        if self.train_batch_size < len(student) or self.train_batch_size % len(student):
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
        if self.teacher_memory + self.reference_memory + self.anchor_memory >= .95:
            raise ValueError('Leave headroom when frozen scoring engines share a teacher GPU')
        safe_name(self.run_name)


def build_verl_plan(experiment, settings=None, loss=None):
    settings, loss = settings or VerlSettings(), loss or TrainingLossConfig.from_experiment(experiment)
    base = Path('outputs/step_4_train/ablation') / PAIR / settings.run_name
    questions = ROOT / experiment['input_data_dir'] / 'questions_source.json'
    split = ROOT / experiment['input_data_dir'] / 'split.json'
    original, partition = read_json(questions), read_json(split)
    if len(original) != 1000 or len(partition['select_ids']) != 700 or len(partition['confirm_ids']) != 300:
        raise ValueError('Preserve the frozen 1000-question 700/300 pool')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (questions, split)}
    plan = {'schema': 'steer_verl_training_v1', 'verl_revision': VERL_REVISION,
            'verl_source': VERL_SOURCE, 'output': str(base), 'settings': asdict(settings),
            'training_loss': asdict(loss), 'student': experiment['student'], 'teacher': experiment['teacher'],
            'tokenizer_paths': experiment['tokenizer_paths'], 'seed': 42, 'input_sha256': hashes,
            'input_data_dir': experiment['input_data_dir'], 'train_split': 'select', 'train_questions': 700,
            'confirm_questions': 300, 'confirm_used_for_training': False,
            'automatic_evaluation': False, 'steering_role': 'teacher', 'direction': 'S-T',
            'granularity': 'global', 'gate': 'all', 'strength_policy': 'fixed',
            'student_rollout_steering': False, 'student_training_steering': False,
            'request_seed_policy': 'constant_42',
            'on_policy': {'ppo_epochs': 1, 'optimizer_steps_per_rollout_batch': 1,
                          'rollout_n': 1, 'temperature': 1., 'top_p': 1., 'top_k': -1},
            'execution_hold': experiment.get('execution_hold'), 'training_started': False,
            'gpu_training_acceptance': 'pending'}
    sources = [*Path(__file__).resolve().parent.glob('*.py'),
               *Path(__file__).resolve().parent.glob('*.yaml'), ROOT/'src/steer_opd/compass_alignment.py',
               ROOT/'src/steer_opd/easysteer_static.py', ROOT/'src/steer_opd/easysteer_worker.py']
    plan['implementation_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(sources)}
    plan=json.loads(json.dumps(plan))
    plan['binding'] = digest(plan)
    return plan


def verl_overrides(plan):
    """Use the pinned public config; teacher Top-K distillation stays disabled."""
    s, root = plan['settings'], ROOT/plan['output']
    return {
        'trainer.use_v1': False, 'model_engine': 'dp',
        'data.train_files': [str(root/'data/train.parquet')],
        'data.val_files': [str(root/'data/confirm.parquet')],
        'data.train_batch_size': s['train_batch_size'], 'data.shuffle': False,
        'data.max_prompt_length': 1024, 'data.max_response_length': s['max_response_length'],
        'data.filter_overlong_prompts': False, 'data.truncation': 'error',
        'data.apply_chat_template_kwargs': {'enable_thinking': False},
        'actor_rollout_ref.model.path': str(ROOT/plan['student']),
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.strategy': 'fsdp',
        'actor_rollout_ref.actor.ppo_mini_batch_size': s['train_batch_size'],
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': s['micro_batch_size'],
        'actor_rollout_ref.actor.ppo_epochs': 1, 'actor_rollout_ref.actor.shuffle': False,
        'actor_rollout_ref.actor.use_dynamic_bsz': False,
        'actor_rollout_ref.actor.optim.lr': s['learning_rate'],
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.batch_invariant': True,
        'actor_rollout_ref.model.use_fused_kernels': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.actor.entropy_coeff': 0., 'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.calculate_entropy': False,
        'actor_rollout_ref.rollout.name': 'vllm', 'actor_rollout_ref.rollout.n': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': s['rollout_memory'],
        'actor_rollout_ref.rollout.temperature': 1., 'actor_rollout_ref.rollout.top_p': 1.,
        'actor_rollout_ref.rollout.top_k': -1, 'actor_rollout_ref.rollout.seed': 42,
        'actor_rollout_ref.rollout.agent.default_agent_loop': 'steer_opd_single_turn',
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(Path(__file__).resolve().parent/'verl_agent.yaml'),
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': s['micro_batch_size'],
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': False,
        'algorithm.adv_estimator': 'reinforce_plus_plus',
        'algorithm.use_kl_in_reward': False, 'algorithm.rollout_correction': None,
        'critic.enable': False, 'reward.reward_model.enable': False, 'distillation.enabled': False,
        'reward.custom_reward_function.path': str(Path(__file__).resolve().parent/'verl_reward.py'),
        'reward.custom_reward_function.name': 'compute_score',
        'trainer.n_gpus_per_node': len(s['student_gpus']), 'trainer.nnodes': 1,
        'trainer.total_epochs': math.ceil(s['total_steps']/max(1,700//s['train_batch_size'])), 'trainer.total_training_steps': s['total_steps'],
        'trainer.val_before_train': False, 'trainer.test_freq': -1,
        'trainer.save_freq': s['save_freq'], 'trainer.max_actor_ckpt_to_keep': 1, 'trainer.logger': ['console'],
        'trainer.project_name': 'steer_opd', 'trainer.experiment_name': s['run_name'],
        'trainer.default_local_dir': str(root/'checkpoints'), 'trainer.resume_mode': 'auto',
        'ray_kwargs.ray_init.num_gpus': len(s['student_gpus']),
    }
