#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec python3 - "$project_dir" "$@" <<'PY'
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import uuid


GIB = 1024 ** 3
DEFAULT_DEST = Path("/home/zhenning/data/working_tmp/steer_opd")
PYTHON_BASE = Path("/home/zhenning/data/miniconda3/envs/verl")

# The receiver extracts into a private directory on the destination filesystem.
# A second SSH call publishes the batch only after both sides of the stream succeed.
RECEIVER = r"""
import filecmp
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


action, work_arg, dest_arg, token, batch_id, kind = sys.argv[1:]
work = Path(work_arg)
dest = Path(dest_arg)
stage = work / ("incoming-" + token)
complete = work / "complete"


def fingerprint(path):
    info = path.lstat()
    result = [info.st_dev, info.st_ino, info.st_mode]
    if not stat.S_ISDIR(info.st_mode):
        result += [info.st_size, info.st_mtime_ns]
        if stat.S_ISLNK(info.st_mode):
            result.append(os.readlink(path))
    return result


def walking_error(error):
    raise error


def merge():
    dest.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(dest.lstat().st_mode):
        raise RuntimeError(f"Destination is not a regular directory: {dest}")
    if stage.stat().st_dev != dest.stat().st_dev:
        raise RuntimeError(f"Staging and destination are on different filesystems: {dest}")
    if kind == "dotenv":
        dotenv = stage / ".env"
        if not stat.S_ISREG(dotenv.lstat().st_mode):
            raise RuntimeError(f"Expected a regular dotenv file: {dotenv}")
        dotenv.chmod(0o600)

    entries = []
    for parent, dirs, files in os.walk(stage, onerror=walking_error, followlinks=False):
        for name in sorted(dirs + files):
            source = Path(parent) / name
            relative = source.relative_to(stage)
            target = dest / relative
            mode = source.lstat().st_mode
            try:
                old_mode = target.lstat().st_mode
            except FileNotFoundError:
                entries.append((source, target, relative, mode, False))
                continue

            same = stat.S_IFMT(mode) == stat.S_IFMT(old_mode)
            if same and stat.S_ISREG(mode):
                same = mode == old_mode and filecmp.cmp(source, target, shallow=False)
            elif same and stat.S_ISLNK(mode):
                same = os.readlink(source) == os.readlink(target)
            if not same:
                raise RuntimeError(f"Destination already has a different entry: {target}")
            entries.append((source, target, relative, mode, True))

    for source, target, relative, mode, exists in entries:
        if exists:
            continue
        if stat.S_ISDIR(mode):
            target.mkdir(mode=stat.S_IMODE(mode))
            target.chmod(stat.S_IMODE(mode))
        elif stat.S_ISLNK(mode):
            target.symlink_to(os.readlink(source))
        else:
            os.link(source, target, follow_symlinks=False)

    record = {
        "dest": str(dest),
        "entries": {str(relative): fingerprint(target) for _, target, relative, _, _ in entries},
    }
    draft = work / ("receipt-" + token)
    draft.write_text(json.dumps(record))
    draft.replace(complete / (batch_id + ".json"))


if action == "prepare":
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    if work.is_symlink() or work.stat().st_uid != os.getuid() or work.stat().st_mode & 0o077:
        raise RuntimeError(f"Migration workspace must be private: {work}")
    complete.mkdir(exist_ok=True)
    if shutil.which("pigz") is None or shutil.which("tar") is None:
        raise RuntimeError("The destination needs pigz and GNU tar")
    for receipt in sorted(complete.glob("*.json")):
        try:
            record = json.loads(receipt.read_text())
            root = Path(record["dest"])
            if all(fingerprint(root / name) == expected for name, expected in record["entries"].items()):
                print(receipt.stem)
        except (OSError, ValueError, KeyError):
            pass
elif action == "extract":
    stage.mkdir(mode=0o700)
    subprocess.run(["tar", "-I", "pigz", "-xpf", "-", "--no-same-owner", "-C", stage], check=True)
elif action == "apply":
    with (work / "merge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merge()
elif action == "clean":
    shutil.rmtree(stage, ignore_errors=True)
    (work / ("receipt-" + token)).unlink(missing_ok=True)
"""


def metadata(info):
    if stat.S_ISDIR(info.st_mode):
        return (info.st_dev, info.st_ino, info.st_mode)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inventory(root, names, excluded):
    entries = {}
    skipped = Counter()
    device = root.stat().st_dev
    pending = list(reversed(names))
    while pending:
        name = pending.pop()
        if name in excluded:
            skipped[name] += 1
            continue
        path = root / name
        info = path.lstat()
        if info.st_dev != device:
            skipped["other filesystem"] += 1
        elif stat.S_ISSOCK(info.st_mode):
            skipped["Unix socket"] += 1
        elif stat.S_ISDIR(info.st_mode):
            entries[name] = info
            pending.extend(name + "/" + child for child in sorted(os.listdir(path), reverse=True))
        elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or stat.S_ISFIFO(info.st_mode):
            entries[name] = info
        else:
            raise RuntimeError(f"Unsupported file type: {path}")
    return entries, skipped


def chunks(entries, budget):
    # Keep all names for each hard-linked file in the same archive.
    groups = {}
    for name, info in entries.items():
        key = (info.st_dev, info.st_ino) if stat.S_ISREG(info.st_mode) and info.st_nlink > 1 else name
        groups.setdefault(key, []).append(name)

    def with_parents(names):
        parents = {str(parent) for name in names for parent in Path(name).parents if str(parent) != "."}
        return sorted(parents.union(names), key=lambda name: (name.count("/"), name))

    batch = []
    size = 0
    for names in groups.values():
        info = entries[names[0]]
        group_size = (info.st_size if stat.S_ISREG(info.st_mode) else 0) + 1024 * len(names)
        if batch and size + group_size > budget:
            yield with_parents(batch), size
            batch = []
            size = 0
        batch.extend(names)
        size += group_size
    if batch:
        yield with_parents(batch), size


def remote_command(host, action, work, dest, token="", batch_id="", kind=""):
    return ["ssh", "-T", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "--", host,
            shlex.join(["python3", "-c", RECEIVER, action, str(work), str(dest), token, batch_id, kind])]


def transfer(root, names, entries, threads, host, work, dest, batch_id, kind):
    token = uuid.uuid4().hex
    receive = remote_command(host, "extract", work, dest, token, batch_id, kind)
    archive = ["tar", "-C", str(root), "--format=posix", "--sparse", "--no-recursion", "--null",
               "-I", f"pigz -1 -p {threads}", "-cf", "-", "-T", "-"]
    try:
        with subprocess.Popen(archive, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as sender:
            with subprocess.Popen(receive, stdin=sender.stdout) as receiver:
                sender.stdout.close()
                sender.communicate(b"".join(os.fsencode(name) + b"\0" for name in names))
                if receiver.wait() != 0 or sender.returncode != 0:
                    raise RuntimeError("The archive stream failed. Run the same command again to resume.")
        for name in names:
            if metadata((root / name).lstat()) != metadata(entries[name]):
                raise RuntimeError(f"Source changed during migration: {root / name}")
        subprocess.run(remote_command(host, "apply", work, dest, token, batch_id, kind), check=True)
    finally:
        cleanup = subprocess.run(remote_command(host, "clean", work, dest, token, batch_id, kind))
        if cleanup.returncode != 0:
            print(f"Could not clean remote staging directory: {work / ('incoming-' + token)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(prog="migrate_to_186.sh", description="Migrate steer_opd in compressed SSH batches")
    parser.add_argument("--run", action="store_true", help="transfer files; the default only prints the local plan")
    parser.add_argument("--batch-gib", type=int, default=8, help="target uncompressed GiB per batch")
    parser.add_argument("--threads", type=int, default=4, help="local pigz threads")
    parser.add_argument("--host", default="zhenning@1.15.156.186")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="absolute destination project path")
    parser.add_argument("--include-python-base", action="store_true", help="also copy the external Conda Python base")
    args = parser.parse_args(sys.argv[2:])
    if args.batch_gib < 1 or args.threads < 1 or not args.dest.is_absolute() or ".." in args.dest.parts:
        parser.error("use positive batch/thread counts and an absolute destination path without '..'")
    if args.dest == args.dest.parent:
        parser.error("the destination must be a project directory")

    backup_paths = {"output_remote", "output", "backups", ".output_backup_versions"}
    sources = [
        ("project", Path(sys.argv[1]), args.dest, None, backup_paths),
        ("dotenv", Path(sys.argv[1]).parent, args.dest.parent, [".env"], set()),
    ]
    if args.include_python_base:
        sources.append(("python", PYTHON_BASE, PYTHON_BASE, None, set()))

    batches = []
    for kind, source, dest, names, excluded in sources:
        entries, skipped = inventory(source, sorted(os.listdir(source)) if names is None else names, excluded)
        if kind == "dotenv" and not stat.S_ISREG(entries[".env"].st_mode):
            raise RuntimeError(f"Expected a regular dotenv file: {source / '.env'}")
        for names, size in chunks(entries, args.batch_gib * GIB):
            identity = json.dumps([str(dest), kind, [(name, metadata(entries[name])) for name in names]])
            batch_id = hashlib.sha256(identity.encode()).hexdigest()
            batches.append((kind, source, dest, names, entries, batch_id, size))
        print(f"{kind}: {len(entries):,} entries at {source}; excluded {dict(skipped)}", flush=True)
    print(f"{len(batches)} batches, {sum(batch[-1] for batch in batches) / GIB:.2f} GiB before compression.", flush=True)

    if not args.run:
        return
    for program in ("tar", "pigz", "ssh"):
        if shutil.which(program) is None:
            raise RuntimeError(f"Required executable not found: {program}")

    work = args.dest.parent / f".{args.dest.name}-migration"
    prepared = subprocess.run(remote_command(args.host, "prepare", work, args.dest),
                              check=True, text=True, stdout=subprocess.PIPE)
    completed = set(prepared.stdout.splitlines())
    for number, (kind, source, dest, names, entries, batch_id, size) in enumerate(batches, 1):
        label = f"[{number}/{len(batches)}] {kind}, {size / GIB:.2f} GiB"
        if batch_id in completed:
            print(f"{label}: verified on destination", flush=True)
        else:
            print(f"{label}: compressing and transferring", flush=True)
            transfer(source, names, entries, args.threads, args.host, work, dest, batch_id, kind)
            print(f"{label}: merged", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
PY
