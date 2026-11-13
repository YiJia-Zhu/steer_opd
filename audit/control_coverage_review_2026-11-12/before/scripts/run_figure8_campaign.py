#!/usr/bin/env python3
"""Persistent single-GPU experiment supervisor; rerun the same command to resume.

Every GPU child goes through fresh idle admission. Never signals any process.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from steer_opd.resumable_core import ROOT, atomic_json, digest, emit, read_json
from steer_opd.campaign_status import supervisor_identity
from steer_opd.generation_runtime import (
    checkpoint_config_hash, default_scheduler_job, scientific_generation_job, prepare_generation_manifest,
)
from steer_opd.method_config import resolve_method, uses_legacy_direction


def relative(path):
    return str(Path(path).relative_to(ROOT))


class Campaign:
    def __init__(self, config_path, resume_paused=False):
        self.config_path = str(Path(config_path).resolve())
        self.cfg = read_json(self.config_path)
        if self.cfg.get("stage") == "pre_opd_overlap_tuning":
            raise ValueError("Use the zero-update diagnostic entry; training campaign is forbidden for this config")
        resolve_method(self.cfg)
        self.run = ROOT / self.cfg["run_dir"]
        state_path = self.run / "campaign.json"
        paused = state_path.exists() and read_json(state_path).get("status") == "paused_by_user"
        if (paused or self.cfg.get("execution_hold")) and not resume_paused:
            raise ValueError("Experiment paused by user; explicit --resume-paused is required after resumption is authorized")
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / "campaign.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not self.cfg["authorized"]:
            raise ValueError("Campaign is not authorized")
        if self.cfg["max_gpus"] != 1:
            raise ValueError("This supervisor is intentionally single-GPU")
        frozen = self.run / "config.json"
        if frozen.exists():
            checkpoint_config_hash(self.cfg)
        else:
            atomic_json(frozen, self.cfg)
        self.data = ROOT / self.cfg["data_dir"]
        runtime_path = self.run / "runtime.json"
        self.runtime = read_json(runtime_path) if runtime_path.exists() else {}
        if self.runtime:
            validation = read_json(self.run / self.runtime["validation"])
            if not validation.get("passed") or validation.get("runtime_hash") != digest(self.runtime):
                raise ValueError("Runtime tuning lacks matching acceptance evidence")
        identity = supervisor_identity(os.getpid())
        self.start_ticks = identity["start_ticks"] if identity else None
        if paused:
            state = read_json(state_path)
            atomic_json(self.run / "pause_history" / f"{time.time_ns()}.json", state)
            atomic_json(state_path, {"status": "resume_requested", "updated_at": time.time()})

    def record_failure(self, exc):
        path = self.run / "campaign.json"
        state = read_json(path) if path.exists() else {}
        if state.get("status") == "paused_by_user":
            return
        state.update({"status": "failed_resumable", "error": repr(exc),
                      "supervisor_pid": os.getpid(), "supervisor_start_ticks": self.start_ticks,
                      "updated_at": time.time()})
        atomic_json(path, state)

    def cpu_stage(self, name, command, *, required=True):
        atomic_json(self.run / "campaign.json", {"status": "running", "phase": name,
                    "supervisor_pid": os.getpid(), "supervisor_start_ticks": self.start_ticks,
                    "updated_at": time.time(), "command": command, "device": "cpu", "max_gpus": 1})
        result = subprocess.run([sys.executable, *command], cwd=ROOT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if result.returncode:
            if required:
                raise subprocess.CalledProcessError(result.returncode, command)
            atomic_json(self.run / "report_error.json", {"phase": name, "exit_code": result.returncode,
                        "updated_at": time.time(), "training_continues": True})
            emit("optional_report_failed", phase=name, exit_code=result.returncode)

    def report_progress(self):
        self.cpu_stage("update_report", ["scripts/figure8_report.py", "--config", self.config_path], required=False)

    def stage(self, name, command):
        for retry in range(3):
            attempt = self.run / "attempts" / f"{name}_{time.time_ns()}"
            atomic_json(self.run / "campaign.json", {"status": "running", "phase": name,
                        "supervisor_pid": os.getpid(), "supervisor_start_ticks": self.start_ticks,
                        "updated_at": time.time(),
                        "attempt": relative(attempt), "command": command, "max_gpus": 1,
                        "resume_command": f".venv-opd/bin/python scripts/run_figure8_campaign.py --config {self.config_path}"})
            emit("stage_start", phase=name, retry=retry)
            cmd = [sys.executable, str(ROOT / "scripts/run_on_idle_gpus.py"), "--count", "1",
                   "--wait-seconds", "43200", "--run-dir", str(attempt), "--", sys.executable, *command]
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode == 0:
                emit("stage_complete", phase=name)
                return
            process = read_json(attempt / "process.json") if (attempt / "process.json").exists() else {}
            log = attempt / "stdout.log"
            tail = log.read_text(errors="replace")[-24000:] if log.exists() else ""
            engine_interrupted = any(marker in tail for marker in ["EngineDeadError", "exit code -9", "exitcode=-9", "signal SIGKILL"])
            if (process.get("exit_code") in [-9, -15, 137, 143] or engine_interrupted) and retry < 2:
                emit("interruption_resume", phase=name, child_exit=process.get("exit_code"))
                time.sleep(10)
                continue
            atomic_json(self.run / "campaign.json", {"status": "failed_resumable", "phase": name,
                        "exit_code": result.returncode, "attempt": relative(attempt), "updated_at": time.time()})
            raise RuntimeError(f"Stage {name} failed; inspect {attempt / 'stdout.log'}; artifacts retained")

    def generate(self, name, model, requests, output, max_tokens, temperature, top_p, vector=None, logprobs=0,
                 capture_logits=False):
        job = {"model": relative(model), "output": relative(output), "requests": requests,
               "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
               "vector": relative(vector) if vector else None, "seed": self.cfg["seed"],
               "gpu_memory_utilization": self.cfg["vllm_gpu_memory_utilization"], "logprobs": logprobs,
               "head_dtype": "float32", "decode_cuda_graph": True,
               "normalization": "native", "backend_version": 6,
               "support_size": self.cfg["shared_vocabulary_size"]}
        if capture_logits:
            job["capture_logits"] = True
        runtime = getattr(self, "runtime", {})
        if runtime:
            job["gpu_memory_utilization"] = runtime["gpu_memory_utilization"]
        if name.endswith("_rollout"):
            if self.cfg.get("student_topk_source") == "rollout":
                job.update({"save_rollout_top_k": self.cfg["supervision_top_k"],
                            "logprobs": self.cfg["supervision_top_k"], "logprobs_mode": "raw_logprobs",
                            "record_token_entropy": True, "token_record_schema": 1})
            else:
                job["replay_audit_tokens"] = 8
            if vector and not uses_legacy_direction(self.cfg):
                method = resolve_method(self.cfg)
                job["method"] = method.to_dict()
                job["per_question_vectors"] = method.granularity == "per_question"
                if method.granularity == "per_question":
                    job["decode_cuda_graph"] = False
                if method.gate != "all":
                    if self.cfg.get("student_topk_source") != "rollout":
                        raise ValueError("Controlled gate branches require saved rollout Top-K")
                    job.update(teacher=self.cfg["teacher"], backend="hf_cached_causal_gate_reference")
            elif not vector and self.cfg.get("steering_method") and resolve_method(self.cfg).gate != "all":
                # Match backend and sampler for ordinary OPD in gated panels.
                # Its policy has no vector, probe or gate intervention.
                method = resolve_method(self.cfg).to_dict()
                method["gate"] = "all"
                job.update(method=method, teacher=self.cfg["teacher"],
                           backend="hf_cached_causal_gate_reference", per_question_vectors=False)
        elif runtime and name.startswith("eval_"):
            job["logprobs"] = None
        job = default_scheduler_job(job)
        if (output / "complete.json").exists():
            if job.get("save_rollout_top_k"):
                _, _, pending = prepare_generation_manifest(output, job)
                if pending:
                    raise ValueError("Completed rollout is missing committed responses")
            return
        path = self.run / "jobs" / f"{name}.json"
        if path.exists():
            existing = read_json(path)
            if scientific_generation_job(existing) != scientific_generation_job(job):
                raise ValueError(f"Generation job changed: {path}")
            if existing != job:
                atomic_json(self.run / "job_history" / f"{name}_{digest(existing)}.json", existing)
        atomic_json(path, job)
        script = ("scripts/figure8_generate_controlled.py" if job.get("backend") == "hf_cached_causal_gate_reference"
                  else "scripts/figure8_generate.py")
        self.stage(name, [script, "--job", str(path)])

    def hf_stage(self, name, stage, group=None, step=None):
        command = ["scripts/figure8_stage.py", "--config", self.config_path, "--stage", stage]
        if group is not None:
            command += ["--group", group, "--step", str(step)]
        self.stage(name, command)

    def checkpoint(self, group, step):
        if step == 0:
            return ROOT / self.cfg["student"]
        return self.run / group / "checkpoints" / f"step_{step:04d}" / "model"

    def acceptance(self):
        out = self.run / "acceptance"
        if not (out / "hf.json").exists():
            self.stage("acceptance_hf", ["scripts/figure8_acceptance.py", "--config", self.config_path, "--phase", "hf"])
        suffix = "_verified"
        for name in ["plain", "steer"]:
            job = read_json(out / f"job_{name}.json")
            job.update({"decode_cuda_graph": True, "head_dtype": "float32", "normalization": "native",
                        "capture_logits": True})
            job = default_scheduler_job(job)
            job["output"] += suffix
            path = out / f"job_{name}{suffix}.json"
            atomic_json(path, job)
            if not (out / (name + suffix) / "complete.json").exists():
                self.stage(f"acceptance_vllm_{name}{suffix}", ["scripts/figure8_generate.py", "--job", str(path)])
        comparison = out / f"backend_comparison{suffix}.json"
        if not comparison.exists() or not read_json(comparison)["passed"]:
            self.stage("acceptance_compare", ["scripts/figure8_acceptance.py", "--config", self.config_path,
                                              "--phase", "compare", "--suffix", suffix])
        if not (out / "backward.json").exists():
            self.stage("acceptance_backward", ["scripts/figure8_acceptance.py", "--config", self.config_path,
                                               "--phase", "backward", "--suffix", suffix])

    def calibration(self):
        if not uses_legacy_direction(self.cfg):
            if resolve_method(self.cfg).extractor == "initial":
                self.prepare_directions(1)
            return
        rows = read_json(self.data / "direction.json")
        for role in ["teacher", "student"]:
            requests = [{"key": r["id"], "id": r["id"], "answer": r["answer"],
                         "prompt_token_ids": r[f"{role}_prompt_ids"], "seed": self.cfg["seed"] + i}
                        for i, r in enumerate(rows)]
            self.generate(f"calibration_{role}", ROOT / self.cfg[role], requests, self.run / "calibration" / role,
                          self.cfg["calibration_max_tokens"], self.cfg["calibration_temperature"], self.cfg["calibration_top_p"])
        if not (self.run / "direction/vectors.pt").exists():
            self.hf_stage("direction_extraction", "extract")

    def prepare_directions(self, step, batch_rows=None):
        """Persist a bank before sampling; never refresh an existing trajectory."""
        from steer_opd.direction_bank import bank_vectors, file_sha256
        import torch

        method = resolve_method(self.cfg)
        extractor_step = method.extraction_step(step)
        root = self.run / "directions" / ("initial" if method.extractor == "initial" else f"step_{extractor_step:04d}")
        target = root / "vectors.pt"
        if target.exists() and (root / "metadata.json").exists():
            metadata = read_json(root / "metadata.json")
            if file_sha256(target) != metadata["vector_file_sha256"]:
                raise ValueError("Frozen direction bank changed")
            return target
        rows = read_json(self.data / "direction.json")
        if method.extractor == "current" and method.granularity == "per_question" and method.refresh_interval == 1:
            rows = list({r["id"]: r for r in (batch_rows or rows)}.values())
        question_path = root / "questions.json"
        atomic_json(question_path, rows)
        directories = {}
        for role in method.roles:
            current_student = role == "student" and method.extractor == "current"
            text_rows = rows if current_student else read_json(self.data / "direction.json")
            output = root / "texts/student" if current_student else self.run / "calibration" / role
            requests = []
            reference_tokenizer = None
            if role == "reference":
                from transformers import AutoTokenizer
                from steer_opd.math_data import prompt_ids
                reference_tokenizer = AutoTokenizer.from_pretrained(ROOT / self.cfg[role], local_files_only=True)
            for index, row in enumerate(text_rows):
                prompt = (prompt_ids(reference_tokenizer, row["question"], enable_thinking=False)
                          if reference_tokenizer is not None else row[f"{role}_prompt_ids"])
                requests.append({"key": row["id"], "id": row["id"], "answer": row["answer"],
                                 "prompt_token_ids": prompt,
                                 "seed": self.cfg["seed"] + index + (extractor_step * 1000000 if current_student else 0)})
            model = self.checkpoint("steer", extractor_step) if current_student else ROOT / self.cfg[role]
            name = f"bootstrap_student_{extractor_step:04d}" if current_student else f"calibration_{role}"
            self.generate(name, model, requests, output, self.cfg["calibration_max_tokens"],
                          self.cfg["calibration_temperature"], self.cfg["calibration_top_p"], logprobs=None)
            directories[role] = relative(output)
        job = {"config": relative(Path(self.config_path)), "questions": relative(question_path),
               "output": relative(root), "method": method.to_dict(), "role_directories": directories,
               "extractor_model": relative(self.checkpoint("steer", extractor_step)),
               "extractor_step": extractor_step,
               "fixed_scales": self.cfg.get("fixed_activation_scales", relative(self.run / "directions/fixed_scales.pt")),
               "cache_token_activations": self.cfg.get("cache_token_activations", self.cfg.get("stage") == "pre_opd_overlap_tuning"),
               "token_activation_dtype": self.cfg.get("token_activation_dtype", "bfloat16"),
               "minimum_pairs": self.cfg.get("minimum_direction_pairs", 32)}
        path = self.run / "jobs" / f"extract_{root.name}.json"
        if self.cfg.get("fixed_activation_scales") and not (ROOT / job["fixed_scales"]).exists():
            raise ValueError("Explicit shared activation scale file is missing")
        atomic_json(path, job)
        command = ["scripts/figure8_extract.py", "--job", str(path)]
        if target.exists():
            self.cpu_stage(f"repair_direction_metadata_{root.name}", command)
        else:
            self.stage(f"direction_{root.name}", command)
        if method.granularity == "per_question":
            bank = torch.load(target, weights_only=True)
            for row in rows:
                bank_vectors(bank, row["id"])
        return target

    def branch_acceptance(self, vector, requests):
        method = resolve_method(self.cfg)
        report = self.run / "branch_acceptance.json"
        if report.exists():
            saved = read_json(report)
            if not saved.get("passed") or saved.get("method_hash") != digest(method.to_dict()):
                raise ValueError("Optional branch lacks matching backend acceptance")
            return
        command = ["scripts/figure8_branch_acceptance.py", "--config", self.config_path,
                   "--bank", relative(vector)]
        if method.gate == "all":
            unique = list({request["id"]: request for request in requests}.values())[:2]
            selected = [{**row, "key": f"branch_{index}"} for index, row in enumerate(unique)]
            output = self.run / "acceptance/bank_backend"
            self.generate("branch_backend_rollout", self.checkpoint("steer", 0), selected, output, 8, 1.0, 1.0,
                          vector, capture_logits=True)
            command += ["--rollouts", relative(output)]
        self.stage("branch_distribution_and_backward_acceptance", command)

    def train_step(self, group, step):
        latest_path = self.run / group / "latest.json"
        if latest_path.exists() and read_json(latest_path)["step"] >= step:
            return
        if ((self.run / group / "steps" / f"{step:04d}" / "train_metrics.json").exists()
                or (self.checkpoint(group, step).parent / "complete.json").exists()):
            return
        rows = read_json(self.data / "opd.json")
        requests = []
        for i in range(self.cfg["prompt_batch_size"]):
            visit = (step - 1) * self.cfg["prompt_batch_size"] + i
            row = rows[visit % len(rows)]
            for replica in range(self.cfg["rollouts_per_prompt"]):
                requests.append({"key": f"{i:03d}_{replica}", "id": row["id"], "answer": row["answer"],
                                 "prompt_token_ids": row["student_prompt_ids"], "visit": visit,
                                 "seed": self.cfg["seed"] + visit * self.cfg["rollouts_per_prompt"] + replica})
        path = self.run / group / "steps" / f"{step:04d}"
        vector = self.run / "direction/vectors.pt" if group == "steer" else None
        if group == "steer" and not uses_legacy_direction(self.cfg):
            from steer_opd.direction_bank import bank_vectors, file_sha256
            import torch
            batch_ids = {r["id"] for r in requests}
            vector = self.prepare_directions(step, [r for r in rows if r["id"] in batch_ids])
            bank = torch.load(vector, weights_only=True)
            sha = file_sha256(vector)
            for request in requests:
                _, vector_id = bank_vectors(bank, request["id"])
                request.update(vector_path=relative(vector), vector_hash=sha, vector_id=vector_id)
            self.branch_acceptance(vector, requests)
        self.generate(f"{group}_{step:04d}_rollout", self.checkpoint(group, step - 1), requests, path / "rollouts",
                      self.cfg["max_response_tokens"], 1.0, 1.0,
                      vector)
        stages = [("teacher", "scores/complete.json")]
        if self.cfg.get("student_topk_source", "hf_replay") == "hf_replay":
            stages.insert(0, ("student", "student_scores/complete.json"))
        for stage, marker in stages:
            if not (path / marker).exists():
                if (stage == "teacher" and self.cfg.get("steering_method")
                        and resolve_method(self.cfg).gate != "all"):
                    self.cpu_stage(f"{group}_{step:04d}_reuse_online_teacher", ["scripts/figure8_stage.py", "--config",
                                   self.config_path, "--stage", "teacher", "--group", group, "--step", str(step)])
                else:
                    self.hf_stage(f"{group}_{step:04d}_{stage}", stage, group, step)
        self.hf_stage(f"{group}_{step:04d}_train", "train", group, step)
        self.report_progress()

    def evaluate(self, group, step):
        name = "base" if step == 0 else f"{group}_{step:04d}"
        out = self.run / "evaluation" / name
        rows = read_json(self.data / "aime2025.json")
        requests = [{"key": f"{i:02d}_{replica:02d}", "id": row["id"], "answer": row["answer"],
                     "prompt_token_ids": row["student_prompt_ids"], "seed": 100000 + i * self.cfg["eval_samples"] + replica}
                    for i, row in enumerate(rows) for replica in range(self.cfg["eval_samples"])]
        self.generate(f"eval_{name}", self.checkpoint(group, step), requests, out / "answers",
                      self.cfg["eval_max_tokens"], self.cfg["eval_temperature"], self.cfg["eval_top_p"])
        if not (out / "accuracy.json").exists():
            # Run potentially slow symbolic grading in separate CPU processes
            # with per-response timeouts; never consumes another GPU.
            self.cpu_stage(f"grade_{name}", ["scripts/figure8_report.py", "--config", self.config_path,
                                            "--grade", str(out)])
        prefixes = read_json(self.data / "dev.json")[:self.cfg["diagnostic_questions"]]
        fixed_requests = [{"key": row["id"], "id": row["id"], "answer": row["answer"],
                           "prompt_token_ids": row["student_prompt_ids"], "seed": 200000 + i}
                          for i, row in enumerate(prefixes)]
        self.generate("fixed_initial_prefixes", ROOT / self.cfg["student"], fixed_requests, self.run / "fixed_prefixes",
                      self.cfg["max_response_tokens"], 1.0, 1.0)
        if not (out / "overlap.json").exists():
            self.hf_stage(f"diagnostic_{name}", "diagnostic", group, step)
        self.report_progress()

    def execute(self):
        self.cpu_stage("prepare_inputs", ["scripts/prepare_figure8_first_run.py", "--config", self.config_path])
        self.acceptance()
        # Establish an actual baseline update before the more expensive offline
        # direction construction. Subsequent comparisons alternate on one GPU.
        self.train_step("vanilla", 1)
        self.calibration()
        for step in range(1, self.cfg["steps"] + 1):
            for group in self.cfg["groups"]:
                self.train_step(group, step)
            if step == 1:
                self.evaluate("vanilla", 0)
            if step in self.cfg["eval_steps"]:
                for group in self.cfg["groups"]:
                    self.evaluate(group, step)
        atomic_json(self.run / "campaign.json", {"status": "complete", "steps_per_group": self.cfg["steps"],
                                                   "finished_at": time.time(), "max_gpus": 1})
        emit("campaign_complete", run_dir=str(self.run))


def execute_with_status(campaign):
    try:
        campaign.execute()
    except BaseException as exc:
        campaign.record_failure(exc)
        raise
    finally:
        campaign.lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_shared_top16_v2.json")
    parser.add_argument("--resume-paused", action="store_true", help="Use only after the user explicitly resumes experiments")
    args = parser.parse_args()
    # Construct outside the handler: failure to acquire another live
    # supervisor's lock must never overwrite that supervisor's state.
    campaign = Campaign(args.config, resume_paused=args.resume_paused)
    execute_with_status(campaign)


if __name__ == "__main__":
    main()
