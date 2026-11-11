"""Render the pinned Rethinking Student Top-K protocol without importing a trainer."""

import copy
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/rethinking_opd.json"


def load_profile(path=PROFILE):
    profile = json.loads(Path(path).read_text())
    if profile["protocol"] != "rethinking_student_topk_reverse_kl_surrogate":
        raise ValueError("Expected Rethinking Student Top-K reverse KL")
    values = profile["paper_hydra_overrides"]
    for key, expected in {
        "algorithm.adv_estimator": "token_reward_direct",
        "+actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode": "student_p",
    }.items():
        if values.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if any(key.startswith("distillation.") for key in values):
        raise ValueError("The newer teacher-top-K distillation backend is not this protocol")
    return profile


def reference_commit(profile):
    return subprocess.check_output(
        ["git", "-C", str(ROOT / profile["reference"]["root"]), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def local_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the project")
    return path


def reference_environment(profile, environ=None):
    env = dict(os.environ if environ is None else environ)
    reference = str(ROOT / profile["reference"]["python_root"])
    excluded = {str(Path(reference).resolve()), str((ROOT / "vendor/verl").resolve())}
    retained = [value for value in env.get("PYTHONPATH", "").split(os.pathsep)
                if value and str(Path(value).resolve()) not in excluded]
    env["PYTHONPATH"] = os.pathsep.join([reference, str(ROOT / "src"), *retained])
    env.pop("STEER_OPD_CONFIG", None)
    env["RAY_ADDRESS"] = "local"
    env["HYDRA_FULL_ERROR"] = "1"
    env.update({"RAY_memory_usage_threshold": "0.99", "CUDA_LAUNCH_BLOCKING": "1",
                "PYTHONUNBUFFERED": "1", "TORCH_NCCL_BLOCKING_WAIT": "1",
                "NCCL_TIMEOUT": "7200", "TORCH_DISTRIBUTED_DEBUG": "INFO"})
    return env


def build_overrides(profile, run_dir, *, steps=None, batch_size=None,
                    response_length=None, n_gpus=None, train_file=None,
                    validation_file=None, launcher_pid=None):
    run_dir = local_output(run_dir)
    values = copy.deepcopy(profile["paper_hydra_overrides"])
    project = profile["project"]
    values.update({
        "data.train_files": str((ROOT / (train_file or project["train_file"])).resolve()),
        "data.val_files": str((ROOT / (validation_file or project["validation_file"])).resolve()),
        "data.seed": project["seed"],
        "+data.apply_chat_template_kwargs.enable_thinking": project["student_apply_chat_template_enable_thinking"],
        "actor_rollout_ref.model.path": str((ROOT / project["student"]).absolute()),
        "reward_model.model.path": str((ROOT / project["teacher"]).absolute()),
        "custom_reward_function.path": str(ROOT / profile["reference"]["python_root"] / "verl/utils/reward_score/ttrl_math/__init__.py"),
        "trainer.logger": ["console"], "trainer.project_name": "steer_opd_rethinking",
        "trainer.experiment_name": run_dir.name,
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.total_training_steps": steps if steps is not None else project["proposed_training_steps"],
        "trainer.resume_mode": "disable", "ray_kwargs.ray_init.num_cpus": 16,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.namespace": "steer_opd_" + run_dir.name,
        "+ray_kwargs.ray_init._temp_dir": f"/proc/{launcher_pid}/cwd/.r" if launcher_pid else str(ROOT / ".r"),
        "+ray_kwargs.ray_init.include_dashboard": False,
        "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH": reference_environment(profile)["PYTHONPATH"],
        "hydra.run.dir": str(run_dir / "hydra"),
    })
    if batch_size is not None:
        values["data.train_batch_size"] = batch_size
        values["actor_rollout_ref.actor.ppo_mini_batch_size"] = batch_size
    if response_length is not None:
        values["data.max_response_length"] = response_length
        values["+actor_rollout_ref.rollout.val_kwargs.max_tokens"] = response_length
        values["actor_rollout_ref.rollout.max_model_len"] = 1024 + response_length
        maximum = max(32768, 1024 + response_length)
        values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] = maximum
        values["actor_rollout_ref.rollout.max_num_batched_tokens"] = maximum
    if n_gpus is not None:
        values["trainer.n_gpus_per_node"] = n_gpus
    return values


def hydra_arguments(values):
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]


def compose_reference(profile, values):
    """CPU-only Hydra composition; never imports Ray, a model or either verl."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = ROOT / profile["reference"]["python_root"] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=hydra_arguments(values))
    return OmegaConf.to_container(config, resolve=False)


def execution_blockers(profile, group="E0", steering_config=None):
    blockers = []
    if not profile.get("execution_authorized", False):
        blockers.append("Experiments remain paused; only code/configuration alignment was authorized.")
    if group != "E0" or steering_config is not None:
        blockers.append("Controlled steering rollout/replay is not implemented in this backend; the old loss-only extension cannot substitute.")
    if not profile["project"]["processed_inputs_prepared"]:
        blockers.append("The repeated 100-question data and AIME2025 parquet are not prepared.")
    return blockers
