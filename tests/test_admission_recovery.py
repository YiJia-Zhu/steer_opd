"""CPU processes simulate admission records; no real GPU is touched."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from steer_opd.resumable_core import ROOT, atomic_json


def script(name):
    spec = importlib.util.spec_from_file_location(f"_admission_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def test_fake_device_admission_records_child_identity_and_prunes_stale_leases(tmp_path, monkeypatch):
    module = script("run_on_idle_gpus")
    monkeypatch.setattr(module, "ROOT", tmp_path.resolve())
    device = {"index": 0, "uuid": "FAKE-NON-CUDA-TEST-DEVICE", "name": "cpu-test", "memory_mib": 0,
              "utilization": 0, "has_compute_process": False, "project_owned": False}
    monkeypatch.setattr(module, "snapshot", lambda: [device])
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    lease_path = tmp_path / ".runtime/gpu_leases.json"
    atomic_json(lease_path, [{"pid": os.getpid(), "start_ticks": -1, "uuids": [device["uuid"]]}])
    attempt = tmp_path / "attempt"
    monkeypatch.setattr(sys, "argv", ["run_on_idle_gpus.py", "--run-dir", str(attempt), "--", sys.executable,
        "-c", "import os,time; print(os.environ['CUDA_VISIBLE_DEVICES']); time.sleep(.1)"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    record = json.loads((attempt / "process.json").read_text())
    assert record["exit_code"] == 0 and record["child_start_ticks"] is not None
    assert (attempt / "stdout.log").read_text().strip() == device["uuid"]
    leases = json.loads(lease_path.read_text())
    assert all(lease["start_ticks"] != -1 for lease in leases)


def test_new_supervisor_adopts_live_waiter_instead_of_starting_duplicate_stage(tmp_path, monkeypatch):
    module = script("run_figure8_campaign")
    monkeypatch.setattr(module, "relative", lambda path: str(path))
    campaign = module.Campaign.__new__(module.Campaign)
    campaign.run, campaign.cfg, campaign.start_ticks = tmp_path, {}, None
    attempt = tmp_path / "attempts/phase_1"
    process_path = attempt / "process.json"
    release = attempt / "release"
    # The CPU child writes the successful process receipt before exiting,
    # exactly as an admitted wrapper does. It never imports CUDA.
    code = ("import time,json,os\nfrom pathlib import Path\n"
            f"while not Path({str(release)!r}).exists():\n    time.sleep(.01)\n"
            f"p=Path({str(process_path)!r}); t=p.with_suffix('.tmp')\n"
            "t.write_text(json.dumps({'exit_code':0})); os.replace(t,p)\n")
    expected = [sys.executable, "-c", code]
    attempt.mkdir(parents=True)
    child = subprocess.Popen(expected, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        atomic_json(attempt / "queue.json", {"pid": child.pid, "start_ticks": ticks(child.pid), "command": expected})
        actual_sleep = time.sleep
        def allow_completion(*args):
            release.touch()
            actual_sleep(.01)
        monkeypatch.setattr(module.time, "sleep", allow_completion)
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("Duplicate stage launch"))
        campaign.stage("phase", ["-c", code])
        assert json.loads(process_path.read_text())["exit_code"] == 0
    finally:
        release.touch()
        child.wait(timeout=5)
