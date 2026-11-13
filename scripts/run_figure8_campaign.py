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
from steer_opd.method_config import resolve_method, uses_legacy_direction, intervention_phases


def relative(path):
    return str(Path(path).relative_to(ROOT))


class Campaign:
    def __init__(self, config_path, resume_paused=False, until_step=None):
        self.config_path = str(Path(config_path).resolve())
        self.cfg = read_json(self.config_path)
        if self.cfg.get("stage") == "pre_opd_overlap_tuning":
            raise ValueError("Use the zero-update diagnostic entry; training campaign is forbidden for this config")
        resolve_method(self.cfg)
        intervention_phases(self.cfg, "steer", 1)
        if self.cfg.get("student_text_refresh", "current") not in ("initial", "current"):
            raise ValueError("student_text_refresh must be initial or current")
        self.run = ROOT / self.cfg["run_dir"]
        state_path = self.run / "campaign.json"
        paused = state_path.exists() and read_json(state_path).get("status") == "paused_by_user"
        if (paused or self.cfg.get("execution_hold")) and not resume_paused:
            raise ValueError("Experiment paused by user; explicit --resume-paused is required after resumption is authorized")
        self.until_step = self.cfg["steps"] if until_step is None else until_step
        if not 1 <= self.until_step <= self.cfg["steps"]:
            raise ValueError("Runtime stop must be within the frozen training budget")
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
        # A killed supervisor may leave its admitted child (or idle waiter)
        # alive. Adopt that exact attempt before launching any duplicate update.
        expected = [sys.executable, *command]
        def live(pid, ticks):
            try:
                path = Path(f"/proc/{pid}")
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                same_identity = ticks is not None and int(fields[19]) == ticks
                if ticks is None:
                    # Read-only adoption of historical receipts predating
                    # start_ticks requires the complete exact child command.
                    argv = [part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part]
                    same_identity = argv == expected
                return same_identity and fields[0] != "Z" and (path / "cwd").resolve(strict=True) == ROOT
            except (OSError, ValueError, IndexError):
                return False
        for old in sorted((self.run / "attempts").glob(f"{name}_*"), reverse=True):
            queue_path, process_path = old / "queue.json", old / "process.json"
            queue = read_json(queue_path) if queue_path.exists() else {}
            process = read_json(process_path) if process_path.exists() else {}
            if (queue.get("command") or process.get("command")) != expected:
                continue
            active = lambda: (live(queue.get("pid"), queue.get("start_ticks")) or
                              live(process.get("pid"), process.get("child_start_ticks")))
            if not active():
                continue
            while active():
                atomic_json(self.run / "campaign.json", {"status": "adopting_existing_stage", "phase": name,
                    "attempt": relative(old), "supervisor_pid": os.getpid(), "supervisor_start_ticks": self.start_ticks,
                    "updated_at": time.time(), "duplicate_update_prevented": True})
                time.sleep(10)
                process = read_json(process_path) if process_path.exists() else {}
            if process.get("exit_code") == 0:
                emit("existing_stage_adopted", phase=name, attempt=relative(old))
                return
        minimum = self.cfg.get("minimum_free_gib", 0) * 1024**3
        if minimum:
            import shutil
            while shutil.disk_usage(ROOT).free < minimum:
                atomic_json(self.run / "campaign.json", {"status": "waiting_project_quota_space", "phase": name,
                    "available_bytes": shutil.disk_usage(ROOT).free, "required_bytes": minimum,
                    "updated_at": time.time(), "supervisor_pid": os.getpid(), "max_gpus": 1})
                time.sleep(10)
        for retry in range(3):
            attempt = self.run / "attempts" / f"{name}_{time.time_ns()}"
            atomic_json(self.run / "campaign.json", {"status": "running", "phase": name,
                        "supervisor_pid": os.getpid(), "supervisor_start_ticks": self.start_ticks,
                        "updated_at": time.time(),
                        "attempt": relative(attempt), "command": command, "max_gpus": 1,
                        "resume_command": getattr(self, "resume_command",
                            f".venv-opd/bin/python scripts/run_figure8_campaign.py --config {self.config_path}")})
            emit("stage_start", phase=name, retry=retry)
            cmd = [sys.executable, str(ROOT / "scripts/run_on_idle_gpus.py"), "--count", "1",
                   "--wait-seconds", os.environ.get("STEER_OPD_GPU_WAIT_SECONDS", "43200"),
                   "--run-dir", str(attempt), "--", sys.executable, *command]
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
                 capture_logits=False, numeric_method=None, ignore_eos=False):
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
        if name.endswith("_rollout") and self.cfg.get("generation_backend") in ("hf_reference", "hf_batched"):
            method = resolve_method(self.cfg).to_dict()
            if not vector:
                method["gate"] = "all"
            job.update(method=method, teacher=self.cfg["teacher"], backend=("hf_batched_causal_gate" if
                       self.cfg["generation_backend"] == "hf_batched" else "hf_cached_causal_gate_reference"),
                       per_question_vectors=bool(vector and method["granularity"] == "per_question"))
        elif name.endswith("_rollout") and self.cfg.get("generation_backend") == "vllm" and job.get("backend"):
            raise ValueError("Exact online gates are not yet integrated into vLLM; no silent backend fallback")
        if numeric_method is not None:
            if not capture_logits or not name.startswith("branch_backend_"):
                raise ValueError("Numeric gate overrides are restricted to full-logit acceptance jobs")
            job.update(method=numeric_method, numeric_gate_override=True)
        if ignore_eos:
            if not capture_logits:
                raise ValueError("EOS override is only for numerical full-logit acceptance")
            job["ignore_eos"] = True
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
        script = {"hf_cached_causal_gate_reference": "scripts/figure8_generate_controlled.py",
                  "hf_batched_causal_gate": "scripts/figure8_generate_batched.py"}.get(job.get("backend"), "scripts/figure8_generate.py")
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
        if self.cfg.get("generation_backend") == "hf_batched" and not (self.run / "plain_batched_acceptance.json").exists():
            rows = read_json(self.data / "opd.json")[:2]
            requests = [{"key": f"plain_{index}", "id": row["id"], "answer": row["answer"],
                         "prompt_token_ids": row["student_prompt_ids"], "seed": self.cfg["seed"]+index}
                        for index, row in enumerate(rows)]
            if len(requests) == 2:
                original = requests[1]["prompt_token_ids"]
                requests[1]["prompt_token_ids"] = (original*((1536+len(original)-1)//len(original)))[:1536]
            output = self.run / "acceptance/plain_batched_backend"
            self.generate("plain_batched_backend_rollout", ROOT / self.cfg["student"], requests, output, 8, 1., 1.,
                          capture_logits=True, ignore_eos=True)
            self.stage("plain_batched_full_support_and_backward", ["scripts/figure8_branch_acceptance.py", "--config",
                self.config_path, "--rollouts", relative(output)])

    def calibration(self):
        if not uses_legacy_direction(self.cfg):
            if resolve_method(self.cfg).extractor == "initial":
                self.prepare_directions(1)
            return
        rows = read_json(ROOT / self.cfg["direction_questions_file"]) if self.cfg.get("direction_questions_file") else read_json(self.data / "direction.json")
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
            bank = torch.load(target, map_location="cpu", weights_only=True)
            if digest(bank["method"]) != digest(method.to_dict()) or digest(metadata["method"]) != digest(method.to_dict()):
                raise ValueError("Cached direction bank method differs from the requested experiment")
            expected_checkpoint = relative(self.checkpoint("steer", extractor_step))
            if (bank["identity"]["step"] != extractor_step
                    or bank["identity"]["checkpoint"] != expected_checkpoint):
                raise ValueError("Cached direction bank extractor checkpoint/step mismatch")
            return target
        rows = (read_json(ROOT / self.cfg["direction_questions_file"]) if self.cfg.get("direction_questions_file")
                else read_json(self.data / "direction.json"))
        if method.extractor == "current" and method.granularity == "per_question" and method.refresh_interval == 1:
            rows = list({r["id"]: r for r in (batch_rows or rows)}.values())
        question_path = root / "questions.json"
        atomic_json(question_path, rows)
        directories = {}
        for role in method.roles:
            current_student = (role == "student" and method.extractor == "current"
                               and self.cfg.get("student_text_refresh", "current") == "current")
            if current_student and extractor_step == 0 and self.cfg.get("reference_pool_dir"):
                current_student = False  # Same frozen S0; reuse the common bootstrap.
            text_rows = rows if current_student else (read_json(ROOT / self.cfg["direction_questions_file"])
                if self.cfg.get("direction_questions_file") else read_json(self.data / "direction.json"))
            shared = ROOT / self.cfg.get("shared_reference_dir", relative(self.run / "calibration"))
            output = root / "texts/student" if current_student else shared / role
            if self.cfg.get("reference_pool_dir") and not current_student:
                output = ROOT / self.cfg["reference_pool_dir"] / role
                if any(not (output / f"{row['id']}.json").exists() for row in text_rows):
                    raise ValueError("Frozen common reference pool is incomplete")
                directories[role] = relative(output)
                continue
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
                                 "seed": (read_json(ROOT / self.cfg["reference_pool_dir"] / "student" / f"{row['id']}.json")["seed"]
                                          if current_student and self.cfg.get("reference_pool_dir") else self.cfg["seed"] + index)})
            model = self.checkpoint("steer", extractor_step) if current_student else ROOT / self.cfg[role]
            name = f"bootstrap_student_{extractor_step:04d}" if current_student else f"calibration_{role}"
            self.generate(name, model, requests, output, self.cfg["calibration_max_tokens"],
                          self.cfg["calibration_temperature"], self.cfg["calibration_top_p"], logprobs=None)
            if current_student:
                output = self.select_bootstrap(root, extractor_step, model, requests, output)
            directories[role] = relative(output)
        job = {"config": relative(Path(self.config_path)), "questions": relative(question_path),
               "output": relative(root), "method": method.to_dict(), "role_directories": directories,
               "extractor_model": relative(self.checkpoint("steer", extractor_step)),
               "extractor_step": extractor_step,
               "fixed_scales": self.cfg.get("fixed_activation_scales", relative(self.run / "directions/fixed_scales.pt")),
               "cache_token_activations": self.cfg.get("cache_token_activations", self.cfg.get("stage") == "pre_opd_overlap_tuning"),
               "token_activation_dtype": self.cfg.get("token_activation_dtype", "bfloat16"),
               "minimum_pairs": self.cfg.get("minimum_direction_pairs", 32)}
        if (extractor_step == 0 and self.cfg.get("shared_initial_pool") and
                self.cfg.get("reference_pool_dir", "").endswith("/references/common")):
            job["shared_initial_pool"] = self.cfg["shared_initial_pool"]
        path = self.run / "jobs" / f"extract_{root.name}.json"
        if self.cfg.get("fixed_activation_scales") and not (ROOT / job["fixed_scales"]).exists():
            raise ValueError("Explicit shared activation scale file is missing")
        atomic_json(path, job)
        command = ["scripts/figure8_extract.py", "--job", str(path)]
        if target.exists():
            self.cpu_stage(f"repair_direction_metadata_{root.name}", command)
        elif job.get("shared_initial_pool"):
            self.cpu_stage(f"direction_{root.name}_from_cache", command)
        else:
            self.stage(f"direction_{root.name}", command)
        if method.granularity == "per_question":
            bank = torch.load(target, weights_only=True)
            for row in rows:
                bank_vectors(bank, row["id"])
        return target

    def select_bootstrap(self, root, extractor_step, model, requests, initial_output):
        from steer_opd.reference_controls import usable_text, BootstrapQualityError
        from steer_opd.direction_bank import file_sha256
        output = root / "texts/student_selected"
        pending = list(requests)
        source = initial_output
        for attempt in range(self.cfg.get("bootstrap_max_attempts", 3)):
            if attempt:
                source = root / f"texts/student_retry{attempt}"
                retry_requests = [{**request, "seed": request["seed"]+attempt*1000000} for request in pending]
                self.generate(f"bootstrap_student_{extractor_step:04d}_retry{attempt}", model, retry_requests, source,
                    self.cfg["calibration_max_tokens"], self.cfg["calibration_temperature"],
                    self.cfg["calibration_top_p"], logprobs=None)
            failed = []
            for request in pending:
                path = source / f"{request['key']}.json"
                record = read_json(path)
                if not usable_text(record):
                    failed.append(request)
                    continue
                chosen = {**record, "bootstrap_selection": {"attempt": attempt,
                    "source": relative(path), "source_sha256": file_sha256(path), "uses_correctness": False}}
                target = output / f"{request['id']}.json"
                if target.exists() and read_json(target) != chosen:
                    raise ValueError("Frozen bootstrap quality selection changed")
                atomic_json(target, chosen)
            pending = failed
            if not pending:
                return output
        atomic_json(root / "bootstrap_quality_failure.json", {"missing_ids": [request["id"] for request in pending],
            "attempts": self.cfg.get("bootstrap_max_attempts", 3), "no_fixed_S0_fallback": True})
        raise BootstrapQualityError("No complete nonrepetitive current-student reference within frozen resampling budget")

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
        if method.gate == "all" or self.cfg.get("generation_backend") == "hf_batched":
            unique = list({request["id"]: request for request in requests}.values())[:2]
            selected = [{**row, "key": f"branch_{index}"} for index, row in enumerate(unique)]
            output = self.run / "acceptance/bank_backend"
            numeric = None
            if self.cfg.get("generation_backend") == "hf_batched" and method.online_gate:
                numeric = {**method.to_dict(), "gate": "matched_position_random", "matching_bin_width": 1,
                           "random_probability": .5, "overlap_threshold": .5}
                for index, request in enumerate(selected):
                    request["seed"] = 700000+index
            self.generate("branch_backend_rollout", self.checkpoint("steer", 0), selected, output, 8, 1.0, 1.0,
                          vector, capture_logits=True, numeric_method=numeric,
                          ignore_eos=self.cfg.get("generation_backend") == "hf_batched")
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
        rollout_steering, learning_steering = intervention_phases(self.cfg, group, step)
        vector = self.run / "direction/vectors.pt" if rollout_steering or learning_steering else None
        if (rollout_steering or learning_steering) and not uses_legacy_direction(self.cfg):
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
        for request in requests:
            request.update(learning_steering=learning_steering,
                           steering_phase=self.cfg.get("steering_phase", "both"))
        self.generate(f"{group}_{step:04d}_rollout", self.checkpoint(group, step - 1), requests, path / "rollouts",
                      self.cfg["max_response_tokens"], 1.0, 1.0,
                      vector if rollout_steering else None)
        stages = [("teacher", "scores/complete.json")]
        if self.cfg.get("student_topk_source", "hf_replay") == "hf_replay":
            stages.insert(0, ("student", "student_scores/complete.json"))
        for stage, marker in stages:
            if not (path / marker).exists():
                if (stage == "teacher" and (self.cfg.get("generation_backend") in ("hf_reference", "hf_batched") or
                        (rollout_steering and self.cfg.get("steering_method")
                        and resolve_method(self.cfg).gate != "all"))):
                    self.cpu_stage(f"{group}_{step:04d}_reuse_online_teacher", ["scripts/figure8_stage.py", "--config",
                                   self.config_path, "--stage", "teacher", "--group", group, "--step", str(step)])
                else:
                    self.hf_stage(f"{group}_{step:04d}_{stage}", stage, group, step)
        if self.cfg.get("loss_objective") in ("compass_candidate", "compass_sampled") and not (path / "auxiliary_scores.json").exists():
            self.hf_stage(f"{group}_{step:04d}_auxiliary", "auxiliary", group, step)
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

    def evaluate_dataset(self, group, step, dataset, samples=1):
        """Named independent test/dev sets; never steer evaluation requests."""
        path = self.data / f"{dataset}.json"
        rows = read_json(path)
        name = f"{dataset}_{group}_{step:04d}"
        output = self.run / "evaluation" / name
        requests = [{"key": f"{index:04d}_{replica:02d}", "id": row["id"], "answer": row["answer"],
                     "prompt_token_ids": row["student_prompt_ids"], "seed": 500000 + index*samples+replica}
                    for index, row in enumerate(rows) for replica in range(samples)]
        max_tokens = (self.cfg.get("dev_eval_max_tokens", self.cfg["max_response_tokens"])
                      if dataset == "selection_dev" else self.cfg["eval_max_tokens"])
        self.generate(f"eval_{name}", self.checkpoint(group, step), requests, output / "answers",
                      max_tokens, self.cfg["eval_temperature"], self.cfg["eval_top_p"])
        if not (output / "accuracy.json").exists():
            self.cpu_stage(f"grade_{name}", ["scripts/figure8_report.py", "--config", self.config_path,
                                           "--grade", str(output)])

    def execute(self):
        self.cpu_stage("prepare_inputs", ["scripts/prepare_figure8_first_run.py", "--config", self.config_path])
        self.acceptance()
        # Establish an actual baseline update before the more expensive offline
        # direction construction. Subsequent comparisons alternate on one GPU.
        if "vanilla" in self.cfg["groups"]:
            self.train_step("vanilla", 1)
        if "steer" in self.cfg["groups"]:
            self.calibration()
        stop = getattr(self, "until_step", self.cfg["steps"])
        matched_target = self.cfg.get("match_training_gpu_seconds_to")
        matched = False
        for step in range(1, stop + 1):
            for group in self.cfg["groups"]:
                self.train_step(group, step)
            if step == 1:
                self.evaluate(self.cfg["groups"][0], 0)
            if step in self.cfg["eval_steps"]:
                for group in self.cfg["groups"]:
                    self.evaluate(group, step)
            if step in self.cfg.get("dev_eval_steps", []):
                for group in self.cfg["groups"]:
                    if not (self.run / f"evaluation/{group}_{step:04d}/overlap.json").exists():
                        self.hf_stage(f"dev_fixed_prefix_{group}_{step}", "diagnostic", group, step)
                    self.evaluate_dataset(group, step, "selection_dev", 1)
            if matched_target and step >= self.cfg.get("minimum_steps", 1):
                from steer_opd.full_experiment import gpu_seconds
                target_seconds = gpu_seconds(ROOT / matched_target)
                if self.cfg.get("match_extra_cost_file"):
                    target_seconds += read_json(ROOT / self.cfg["match_extra_cost_file"])["additional_main_gpu_seconds"]
                if target_seconds <= 0:
                    raise ValueError("Compute matching needs a finished reference with measured training GPU seconds")
                if gpu_seconds(self.run) >= target_seconds:
                    stop, matched = step, True
                    break
        if matched_target:
            for group in self.cfg["groups"]:
                self.evaluate(group, stop)
        finished = stop == self.cfg["steps"] or matched
        for dataset in (self.cfg.get("additional_evaluation_sets", []) if finished else []):
            for group in self.cfg["groups"]:
                self.evaluate_dataset(group, stop, dataset, 1)
        status = "complete" if finished else "checkpointed_resumable"
        if matched_target and not matched:
            status = "maximum_steps_reached_compute_unmatched"
        atomic_json(self.run / "campaign.json", {"status": status, "compute_matched": matched,
                                                   "cost_scope": ("main training plus measured setup/model-selection; final benchmarks excluded"
                                                       if self.cfg.get("match_extra_cost_file") else "training stages including loading; excludes evaluation and shared reference generation"),
                                                   "steps_per_group": stop,
                                                   "finished_at": time.time(), "max_gpus": 1})
        emit("campaign_complete" if status == "complete" else "campaign_checkpointed", run_dir=str(self.run), steps=stop)


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
    parser.add_argument("--until-step", type=int, help="Runtime checkpoint boundary; frozen total budget is unchanged")
    args = parser.parse_args()
    # Construct outside the handler: failure to acquire another live
    # supervisor's lock must never overwrite that supervisor's state.
    campaign = Campaign(args.config, resume_paused=args.resume_paused, until_step=args.until_step)
    execute_with_status(campaign)


if __name__ == "__main__":
    main()
