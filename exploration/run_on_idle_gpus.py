#!/usr/bin/env python3
"""Start a project command after two consecutive idle-GPU observations.

This is admission checking, not a machine-wide resource reservation. Cleanup
targets only its own command's session. GPU UUIDs, PID and exit code are recorded.
"""

import argparse
import csv
import codecs
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import sys
import signal


ROOT = Path('/local_nvme/data/zhenning/working_tmp/steer_opd')


def process_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z" or Path(f"/proc/{pid}/cwd").resolve(strict=True) != ROOT/'exploration':
            return None
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_snapshot():
    files = sorted([*(ROOT / "src/steer_opd").glob("*.py"), *(ROOT / "scripts").glob("*.py"), *(ROOT / "scripts").glob("*.sh")])
    bodies = {str(path.relative_to(ROOT)): path.read_text() for path in files}
    hashes = {path: hashlib.sha256(body.encode()).hexdigest() for path, body in bodies.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = ROOT / ".runtime/source_revisions" / identity / "source_bundle.json"
    if not target.exists():
        write_json(target, {"sha256": hashes, "source": bodies})
    return {"id": identity, "bundle": str(target.relative_to(ROOT))}


def snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True)
    processes = [(row[0].strip(), int(row[1])) for row in csv.reader(io.StringIO(apps)) if len(row) >= 2]
    occupied = {uuid for uuid, pid in processes}
    project_owned = {uuid for uuid, pid in processes if process_ticks(pid) is not None}
    rows = []
    for row in csv.reader(io.StringIO(output)):
        index, uuid, name, memory, utilization, driver = [value.strip() for value in row]
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_mib": int(memory), "utilization": int(utilization),
                     "has_compute_process": uuid in occupied, "project_owned": uuid in project_owned, "driver_version":driver})
    return rows


def idle(row):
    return not row["has_compute_process"] and row["memory_mib"] <= 128 and row["utilization"] <= 1


def parse_gpu_ids(value):
    if value is None:return None
    try:ids=[int(x.strip()) for x in value.split(',')]
    except ValueError:raise ValueError('GPU IDs must be comma-separated physical indices') from None
    if not ids or any(i<0 for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('GPU IDs must be unique nonnegative physical indices')
    return ids


def select_idle(first,second,leased,count,gpu_ids=None):
    rows={r['index']:r for r in second}
    ids=list(rows) if gpu_ids is None else gpu_ids
    if set(ids)-set(rows):raise ValueError(f'Unknown physical GPU IDs: {set(ids)-set(rows)}')
    earlier={r['uuid'] for r in first if idle(r)}
    return [rows[i] for i in ids if idle(rows[i]) and rows[i]['uuid'] in earlier
            and rows[i]['uuid'] not in leased][:count]


def tee_process(child,log,console=None):
    """Drain without line buffering: tqdm carriage-return updates remain live."""
    console=sys.stdout if console is None else console
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    while True:
        block=os.read(child.stdout.fileno(),65536)
        if not block: break
        if log is not None:log.write(block);log.flush()
        text=decoder.decode(block)
        try: console.write(text);console.flush()
        except (BrokenPipeError,OSError): console=None
        if console is None:
            class Quiet:
                def write(self,*args): pass
                def flush(self): pass
            console=Quiet()
    tail=decoder.decode(b'',final=True)
    try: console.write(tail);console.flush()
    except (BrokenPipeError,OSError): pass
    return child.wait()


def stop_owned_group(child,start_ticks,grace=10):
    """Finish cleanup even if a second Ctrl+C/TERM arrives; never follow PID reuse."""
    if start_ticks is None:return
    try:
        proc=Path(f'/proc/{child.pid}')
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        if int(fields[19])!=start_ticks:return
        if fields[0]!='Z' and ((proc/'cwd').resolve()!=ROOT or os.getpgid(child.pid)!=child.pid or os.getsid(child.pid)!=child.pid):
            raise RuntimeError('Refusing to stop a process outside its own session')
    except FileNotFoundError:
        # An exited leader may leave workers. Only its original group can
        # retain this PGID, and the scan below checks project ownership.
        members=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                if (fields[0]!='Z' and int(fields[2])==child.pid and int(fields[3])==child.pid
                        and int(fields[19])>=start_ticks and process_ticks(int(proc.name)) is not None):members.append(proc)
            except OSError:pass
        if not members:return
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in handlers:signal.signal(s,signal.SIG_IGN)
    try:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=grace)
        except subprocess.TimeoutExpired:pass
        # The leader can exit before its EngineCore; identify surviving members.
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                own=(fields[0]!='Z' and int(fields[2])==child.pid and int(fields[3])==child.pid
                     and process_ticks(int(path.name)) is not None)
            except (OSError,ValueError,IndexError):continue
            if own:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
        child.wait()
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)


def main():
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--max-project-gpus", type=int, default=2)
    parser.add_argument("--gpu-ids",help="Only these physical nvidia-smi indices, in preference order")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.count <= args.max_project_gpus <= 2:
        parser.error("A command is required; project GPU limit is1 or2")
    gpu_ids=parse_gpu_ids(args.gpu_ids)
    if gpu_ids is not None and len(gpu_ids)<args.count:
        parser.error('Selected GPU pool is smaller than the requested tensor parallelism')
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Run directory must be in the project")
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "process.json"
    if record_path.exists():
        raise FileExistsError("Use a new run directory; never overwrite an existing process record")
    queued_monotonic=time.monotonic()
    queue = {"pid": os.getpid(), "start_ticks": process_ticks(os.getpid()), "command": command,
             "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "waiting_idle_gpu", "gpu_ids":gpu_ids}
    write_json(run_dir / "queue.json", queue)
    lease_path = ROOT / "exploration/.runtime/gpu_leases.json"
    lock_path = ROOT / "exploration/.runtime/gpu_admission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            leases = json.loads(lease_path.read_text()) if lease_path.exists() else []
            leases = [lease for lease in leases if process_ticks(lease["pid"]) == lease["start_ticks"] and lease["start_ticks"] is not None]
            first = snapshot()
            time.sleep(2)
            second = snapshot()
            leased = {uuid for lease in leases for uuid in lease["uuids"]}
            own = leased | {row["uuid"] for row in second if row.get("project_owned")}
            selected = select_idle(first,second,leased,args.count,gpu_ids)
            if len(own)+args.count > args.max_project_gpus:
                selected = []
            if len(selected) == args.count:
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(r["uuid"] for r in selected)
                env["STEER_OPD_GPU_ADMISSION"] = str(record_path)
                env["PYTHONHASHSEED"]="42"
                env["PYTHONUNBUFFERED"]="1"
                record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "cwd": str(ROOT),
                    "command": command, "gpus": selected, "before": [first, second], "supervisor_pid": os.getpid(),
                    "queue_wait_seconds":time.monotonic()-queued_monotonic}
                record["source_revision"] = source_snapshot()
                log = (run_dir / "stdout.log").open("xb")
                try:
                    child = subprocess.Popen(command, cwd=ROOT/'exploration', env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record.update(pid=child.pid, child_start_ticks=process_ticks(child.pid))
                write_json(record_path, record)
                leases.append({"pid": child.pid, "start_ticks": record["child_start_ticks"],
                    "uuids": [row["uuid"] for row in selected], "process_record": str(record_path)})
                write_json(lease_path, leases)
                queue.update(status="launched", child_pid=child.pid)
                write_json(run_dir / "queue.json", queue)
                break
            write_json(lease_path, leases)
        if time.monotonic() >= deadline:
            (run_dir / "no_idle_gpu.json").write_text(json.dumps(second, indent=2) + "\n")
            raise SystemExit("Not enough idle GPUs; no process was started or terminated")
        print("Waiting for idle GPUs; no external processes are changed", flush=True)
        time.sleep(10)
    try:
        print(json.dumps({"pid": child.pid, "gpus": selected, "log": str(run_dir / "stdout.log")}), flush=True)
        child_started=time.monotonic()
        try:
            record["exit_code"]=tee_process(child,log)
        except BaseException as error:
            record["error"]=repr(error)
            stop_owned_group(child,record['child_start_ticks'])
            record["exit_code"]=child.returncode
            raise
        finally:
            record["elapsed_seconds"]=time.monotonic()-child_started
            record["finished_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(record_path,record)
    finally:
        log.close()
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(record_path, record)
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
