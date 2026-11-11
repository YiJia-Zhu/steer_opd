#!/usr/bin/env python3
"""Preview pinned Rethinking Student Top-K defaults; never start Ray in preview.

The former forward_kl_topk_tail launcher is archived under
 audit/rethinking_alignment_2026-11-12/before/scripts/launch_pilot.py.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from steer_opd.rethinking_protocol import (
    PROFILE, ROOT, build_overrides, compose_reference, execution_blockers,
    hydra_arguments, load_profile, local_output, reference_commit, reference_environment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--group", choices=["E0", "E3", "E4"], default="E0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steering-config", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--response-length", type=int)
    parser.add_argument("--n-gpus", type=int)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.compose_only and args.execute:
        parser.error("--compose-only and --execute are mutually exclusive")
    profile = load_profile(args.profile)
    actual_commit = reference_commit(profile)
    if actual_commit != profile["reference"]["commit"]:
        raise RuntimeError("Reference fork commit differs from the audited protocol")
    run_dir = local_output(args.run_dir)
    blockers = execution_blockers(profile, args.group, args.steering_config)
    if args.execute and blockers:
        raise RuntimeError("Execution blocked: " + " ".join(blockers))
    if args.execute and not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    values = build_overrides(
        profile, run_dir, steps=args.steps, batch_size=args.batch_size,
        response_length=args.response_length, n_gpus=args.n_gpus,
        train_file=args.train_file, validation_file=args.validation_file,
        launcher_pid=os.getpid() if args.execute else None,
    )
    composed = compose_reference(profile, values)
    command = [sys.executable, "-m", profile["reference"]["training_entry"], *hydra_arguments(values)]
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "protocol": profile["protocol"], "reference_commit": actual_commit,
              "group": args.group, "preview_only": not args.execute,
              "execution_blockers": blockers, "runtime_gpu_validated": False,
              "command": command, "hydra_overrides": values,
              "source_python_root": profile["reference"]["python_root"],
              "scope_exceptions": profile["scope_exceptions"]}
    with (run_dir / "launch.json").open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    (run_dir / "composed_config.json").write_text(json.dumps(composed, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"preview_only": True, "protocol": profile["protocol"],
                          "output": str(run_dir), "execution_blockers": blockers}, ensure_ascii=False, indent=2))
        return
    for key in ("data.train_files", "data.val_files"):
        if not Path(values[key]).is_file():
            raise FileNotFoundError(values[key])
    (ROOT / ".r").mkdir(exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, env=reference_environment(profile))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
