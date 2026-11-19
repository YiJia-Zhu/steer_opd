"""Pinned Rethinking-OPD runtime defaults, with explicit project adaptations."""
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT,read_json

REVISION='ac26e38d6f1572eb027597b48a9f4e01f6915ef8'
REFERENCE='ref/implementation_sources/rethinking_comparison.json'
REFERENCE_SHA256='7906f627617775417b8da9c24d00abeb208b8714a1c6797c5198ba9d0d40c618'


def defaults():
    path=ROOT/REFERENCE
    if file_sha256(path)!=REFERENCE_SHA256:raise ValueError('Pinned Rethinking configuration evidence changed')
    packet=read_json(path)
    if packet['reference_commit']!=REVISION:raise ValueError('Rethinking source revision changed')
    return {v['key']:v['reference'] for v in packet['paper_default_checks']}


def inherited_overrides():
    values=defaults()
    keys=[
        'data.shuffle','data.max_prompt_length','data.max_response_length','data.filter_overlong_prompts',
        'data.truncation','data.return_raw_chat','actor_rollout_ref.model.use_remove_padding',
        'actor_rollout_ref.model.enable_activation_offload','actor_rollout_ref.model.enable_gradient_checkpointing',
        'actor_rollout_ref.actor.strategy','actor_rollout_ref.actor.optim.lr',
        'actor_rollout_ref.actor.optim.optimizer','actor_rollout_ref.actor.optim.betas',
        'actor_rollout_ref.actor.optim.weight_decay','actor_rollout_ref.actor.optim.lr_scheduler_type',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio','actor_rollout_ref.actor.ppo_epochs',
        'actor_rollout_ref.actor.use_dynamic_bsz','actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu','actor_rollout_ref.actor.ulysses_sequence_parallel_size',
        'actor_rollout_ref.actor.use_kl_loss','actor_rollout_ref.actor.loss_agg_mode',
        'actor_rollout_ref.actor.entropy_coeff','actor_rollout_ref.actor.grad_clip',
        'actor_rollout_ref.actor.clip_ratio','actor_rollout_ref.actor.clip_ratio_low',
        'actor_rollout_ref.actor.clip_ratio_high','actor_rollout_ref.actor.clip_ratio_c',
        'actor_rollout_ref.actor.fsdp_config.forward_prefetch','actor_rollout_ref.actor.fsdp_config.model_dtype',
        'actor_rollout_ref.rollout.name','actor_rollout_ref.rollout.temperature',
        'actor_rollout_ref.rollout.top_p','actor_rollout_ref.rollout.top_k',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz','actor_rollout_ref.rollout.tensor_model_parallel_size',
        'actor_rollout_ref.rollout.max_model_len','actor_rollout_ref.rollout.max_num_batched_tokens',
        'actor_rollout_ref.rollout.repetition_penalty','actor_rollout_ref.rollout.calculate_log_probs',
        'actor_rollout_ref.rollout.val_kwargs.do_sample','actor_rollout_ref.rollout.val_kwargs.n',
        'actor_rollout_ref.rollout.val_kwargs.temperature','actor_rollout_ref.rollout.val_kwargs.top_p',
        'trainer.val_before_train','trainer.log_val_generations','trainer.test_freq']
    # Same values, with one renamed field in the newer pinned VERL schema.
    return {key:values[key] for key in keys}


def provenance():
    return {'revision':REVISION,'repository':'https://github.com/thunlp/OPD',
        'reference_file':REFERENCE,'reference_sha256':REFERENCE_SHA256,
        'script_sha256':'8377cb1b7bec6ebf380f8274623ce0188591ce90c95216ccd85b6d7aa36974a1',
        'adaptations':{
            'models_data_and_loss':'Use the selected Qwen pair and one 1024-question pool for SV, overlap and OPD; S-T teacher SV and two discussed losses; no native-ID TopK KL or PPO clipping.',
            'gpu_count':'Shared two-card pool, student FSDP ranks (strategy=fsdp) and rollout TP1; teacher EasySteer TP2.',
            'old_logprobs':'Reuse raw model scores at vLLM sampled tokens through the old-policy bypass. Presence-penalized sampling is distinct from these raw scores; retain sampled raw-model OPD and log actor/raw-rollout differences. No exact unpenalized on-policy KL claim or automatic PPO/IS/RS/centering.',
            'offload':'Explicit CPU offload and sleep acknowledgements are required for two-card sharing; actor param/optimizer offload=True instead of official False.',
            'teacher_dtype':'Preserve BF16 EasySteer teacher, not official FP32 FSDP teacher.',
            'training_budget':'Use 32 questions x4 responses per step, at most 64 steps (two epochs), and evaluate/save every16 steps plus final; also evaluate step0.',
            'generation_length':'User 2026-11-18 efficiency choice: max response length16384 for training rollouts and validation, instead of the official7168; prompt1024, student context17408, frozen scoring context24576.',
            'seed_and_logging':'Global seed42; each training question uses rollout seeds42,43,44,45, reused across questions and steps and assigned before worker dispatch; evaluation requests keep42. Local console; no global Ray stop or automatic remote logging.',
            'compatibility':'Use current pinned VERL public loss hook; actor gradient is the discussed sampled OPD surrogate, not the Rethinking TopK/PPO objective.',
            'checkpoint_retention':'Keep every saved model; only the latest save retains optimizer/scheduler/RNG/dataloader state. Save every16 steps/final per user choice instead of the official20. Durable full and model receipts precede old-state retirement; disable upstream eager pruning.',
            'storage_retention':'minimal_v1 retains current-batch inputs/scores until a successful update, then retires them; keep scalar monitoring and disable rollout/validation text archives.'}}
