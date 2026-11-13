"""Read Linux process identity without sending signals or initializing CUDA."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def supervisor_identity(pid, proc_root=Path("/proc")):
    if not isinstance(pid, int) or pid <= 0:
        return None
    path = proc_root / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (path / "cmdline").read_bytes().split(b"\0")
        entries = (b"scripts/run_figure8_campaign.py", b"scripts/run_pre_opd_overlap_tuning.py", b"scripts/run_full_scheme.py")
        if not any(part.endswith(entries) for part in command):
            return None
        if (path / "cwd").resolve(strict=True) != ROOT:
            return None
        return {"pid": pid, "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def supervisor_is_alive(record, proc_root=Path("/proc")):
    identity = supervisor_identity(record.get("supervisor_pid"), proc_root)
    if identity is None:
        return False
    expected = record.get("supervisor_start_ticks")
    return expected is None or expected == identity["start_ticks"]
