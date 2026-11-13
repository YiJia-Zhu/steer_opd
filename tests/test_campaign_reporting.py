"""Regressions for the real missing-plot dependency and stale running state."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess

import pytest

from steer_opd.campaign_status import ROOT, supervisor_is_alive


def script(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_plot_dependency_preserves_metrics_and_does_not_fail_report(tmp_path, monkeypatch):
    module = script("figure8_report")
    evaluation = tmp_path / "evaluation/base"
    evaluation.mkdir(parents=True)
    accuracy = {"accuracy": 5 / 480, "mean_tokens": 13325.4, "truncated_fraction": 0.398}
    original = json.dumps(accuracy)
    (evaluation / "accuracy.json").write_text(original)

    def missing_plot(*args):
        raise ModuleNotFoundError("No module named 'matplotlib'")

    monkeypatch.setattr(module, "plot_curves", missing_plot)
    module.report({"run_dir": str(tmp_path), "groups": ["vanilla", "steer"], "steps": 50, "eval_steps": [0, 10]})
    assert (evaluation / "accuracy.json").read_text() == original
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "curves.csv").exists()
    result = json.loads((tmp_path / "results.json").read_text())
    assert len(result["evaluation"]) == 2
    assert json.loads((tmp_path / "plot_status.json").read_text())["status"] == "deferred"


def test_cpu_exception_records_failed_resumable_and_releases_lock(tmp_path):
    module = script("run_figure8_campaign")
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run = tmp_path
    campaign.start_ticks = 123
    campaign.lock = io.StringIO()
    (tmp_path / "campaign.json").write_text(json.dumps({"status": "running", "phase": "grade_base"}))

    def failure():
        raise subprocess.CalledProcessError(1, ["figure8_report.py", "--grade"])

    campaign.execute = failure
    with pytest.raises(subprocess.CalledProcessError):
        module.execute_with_status(campaign)
    result = json.loads((tmp_path / "campaign.json").read_text())
    assert result["status"] == "failed_resumable"
    assert result["phase"] == "grade_base"
    assert "CalledProcessError" in result["error"]
    assert campaign.lock.closed


def test_process_identity_rejects_absent_reused_and_zombie_supervisors(tmp_path):
    record = {"supervisor_pid": 321, "supervisor_start_ticks": 1234}
    assert not supervisor_is_alive(record, tmp_path)
    proc = tmp_path / "321"
    proc.mkdir()
    proc.joinpath("cmdline").write_bytes(b"python\0scripts/run_figure8_campaign.py\0")
    proc.joinpath("cwd").symlink_to(ROOT, target_is_directory=True)
    fields = ["S", *(["0"] * 18), "1234"]
    proc.joinpath("stat").write_text("321 (python) " + " ".join(fields))
    assert supervisor_is_alive(record, tmp_path)
    assert not supervisor_is_alive({**record, "supervisor_start_ticks": 1235}, tmp_path)
    fields[0] = "Z"
    proc.joinpath("stat").write_text("321 (python) " + " ".join(fields))
    assert not supervisor_is_alive(record, tmp_path)


def test_optional_report_subprocess_failure_does_not_abort_training(tmp_path, monkeypatch):
    module = script("run_figure8_campaign")
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run = tmp_path
    campaign.start_ticks = 123
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 1))
    campaign.cpu_stage("update_report", ["figure8_report.py"], required=False)
    result = json.loads((tmp_path / "report_error.json").read_text())
    assert result["training_continues"]


def test_runtime_tuning_preserves_existing_generation_job_and_scientific_parameters(tmp_path, monkeypatch):
    module = script("run_figure8_campaign")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run = tmp_path
    campaign.cfg = {"seed": 42, "vllm_gpu_memory_utilization": 0.75, "shared_vocabulary_size": 151665}
    campaign.runtime = {"gpu_memory_utilization": 0.9, "scheduler": "vllm_defaults"}
    campaign.stage = lambda *args: None
    model, output = tmp_path / "model", tmp_path / "answers"
    requests = [{"key": "sample", "prompt_token_ids": [1], "seed": 42}]
    campaign.generate("steer_0009_rollout", model, requests, output, 7168, 1.0, 1.0)
    path = tmp_path / "jobs/steer_0009_rollout.json"
    original = path.read_bytes()
    job = json.loads(original)
    assert "max_num_seqs" not in job and "batch_size" not in job
    assert job["scheduler"] == "vllm_defaults"
    assert job["replay_audit_tokens"] == 8
    assert job["max_tokens"] == 7168 and job["support_size"] == 151665
    legacy = {**job, "max_num_seqs": 16, "batch_size": 32}
    path.write_text(json.dumps(legacy))
    campaign.generate("steer_0009_rollout", model, requests, output, 7168, 1.0, 1.0)
    assert path.read_bytes() == original
    assert len(list((tmp_path / "job_history").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="Generation job changed"):
        campaign.generate("steer_0009_rollout", model, requests, output, 7168, 0.7, 1.0)
    campaign.generate("eval_steer_0010", model, requests, output, 31744, 0.7, 0.95)
    evaluation = json.loads((tmp_path / "jobs/eval_steer_0010.json").read_text())
    assert "max_num_seqs" not in evaluation and evaluation["logprobs"] is None
    assert evaluation["max_tokens"] == 31744 and evaluation["top_p"] == 0.95
