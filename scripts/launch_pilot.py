#!/usr/bin/env python3
"""Project-local verl launcher; never attaches to/stops an existing Ray cluster."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["E0", "E3", "E4"], default="E0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steering-config", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--response-length", type=int, default=2048)
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    verify_runtime()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Run directory must stay in project")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.compose_only and not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    if args.group != "E0" and not args.steering_config:
        raise ValueError("E3/E4 require a validated, fixed steering config")
    if args.group == "E0" and args.steering_config:
        raise ValueError("E0 must have no steering config")
    env = dict(os.environ)
    env.pop("STEER_OPD_CONFIG", None)
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_ADDRESS"] = "local"
    # A short alias to a directory physically INSIDE this project avoids Linux's
    # Unix socket path-length limit. This launcher stays alive while Ray runs.
    ray_dir = ROOT / ".r"
    ray_dir.mkdir(exist_ok=True)
    short_ray_dir = f"/proc/{os.getpid()}/cwd/.r"
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": "False",
        "data.train_files": str(ROOT / "data/math500_pilot_v2/train_pilot.parquet"),
        "data.val_files": str(ROOT / "data/math500_pilot_v2/dev_pilot.parquet"),
        "data.train_batch_size": args.batch_size, "data.max_prompt_length": 1024,
        "data.max_response_length": args.response_length, "data.filter_overlong_prompts": "True",
        "data.truncation": "error", "data.shuffle": "True", "data.seed": 42,
        "actor_rollout_ref.model.path": "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B",
        "actor_rollout_ref.model.use_remove_padding": "False",
        "actor_rollout_ref.model.enable_gradient_checkpointing": "True",
        "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.use_torch_compile": "False",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": "False",
        "actor_rollout_ref.actor.fsdp_config.param_offload": "False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "False",
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.checkpoint.save_contents": '["model","extra","hf_model"]',
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
        "actor_rollout_ref.rollout.enforce_eager": "True",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.max_model_len": 1024 + args.response_length + 1,
        "actor_rollout_ref.rollout.max_num_seqs": 32,
        "actor_rollout_ref.rollout.agent.num_workers": 2,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "r1_native_single_turn",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(ROOT / "configs/r1_agent_loop.yaml"),
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "trainer.logger": '["console"]', "trainer.project_name": "steer_opd_math500",
        "trainer.experiment_name": run_dir.name, "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1, "trainer.val_before_train": "False", "trainer.test_freq": -1,
        "trainer.total_epochs": 100, "trainer.total_training_steps": args.steps,
        "trainer.save_freq": min(20, args.steps), "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.resume_mode": "disable", "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "distillation.enabled": "True", "distillation.n_gpus_per_node": 1,
        "distillation.nnodes": 1,
        "distillation.teacher_models.teacher_model.model_path": "/home/zhenning/data/huggingface_datasets/JustRL-DeepSeek-1.5B",
        "distillation.teacher_models.teacher_model.inference.name": "vllm",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": 1,
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization": 0.65,
        "distillation.teacher_models.teacher_model.inference.max_model_len": 1024 + args.response_length + 1,
        "distillation.teacher_models.teacher_model.inference.max_num_seqs": 32,
        "distillation.distillation_loss.loss_mode": "forward_kl_topk_tail",
        "distillation.distillation_loss.topk": 16,
        "distillation.distillation_loss.use_policy_gradient": "False",
        "distillation.distillation_loss.use_task_rewards": "False",
        "distillation.distillation_loss.loss_max_clamp": "null",
        "distillation.distillation_loss.log_prob_min_clamp": "null",
        "reward.custom_reward_function.path": str(ROOT / "src/steer_opd/reward.py"),
        "reward.custom_reward_function.name": "compute_score",
        "ray_kwargs.ray_init.num_cpus": 16,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.namespace": "steer_opd_" + run_dir.name,
        "+ray_kwargs.ray_init._temp_dir": short_ray_dir,
        "+ray_kwargs.ray_init.include_dashboard": "False",
        "ray_kwargs.ray_init.runtime_env.py_executable": "null",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units": 2,
        "hydra.run.dir": str(run_dir / "hydra"),
    }
    if args.steering_config:
        config_path = args.steering_config.resolve()
        config = json.loads(config_path.read_text())
        if config["group"] != args.group:
            raise ValueError("Steering group/config mismatch")
        env["STEER_OPD_CONFIG"] = str(config_path)
        values["actor_rollout_ref.model.external_lib"] = "steer_opd.verl_extension"
        values["+ray_kwargs.ray_init.runtime_env.env_vars.STEER_OPD_CONFIG"] = str(config_path)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *[f"{k}={v}" for k, v in values.items()]]
    if args.compose_only:
        command += ["--cfg", "job"]
    (run_dir / "launch.json").write_text(json.dumps({
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "group": args.group,
        "command": command, "ray_storage": str(ray_dir), "compose_only": args.compose_only,
        "student_rollout_steering": False, "resume_supported": False,
    }, indent=2) + "\n")
    result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
