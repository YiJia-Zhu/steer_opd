"""Training capacity checks include visible per-user block quotas."""
import json
from pathlib import Path
import re
import shutil
import subprocess


def quota_remaining(output):
    """Parse quota -w -v block counts, whose unit is KiB (without -s)."""
    for line in output.splitlines():
        match = re.match(r'^\s*\S+\s+(\d+)\*?\s+(\d+)\s+(\d+)(?:\s|$)', line)
        if match:
            used, soft, hard = map(int, match.groups())
            limits = [n for n in (soft, hard) if n]
            return max(0, min(limits)-used)*1024 if limits else None
    return None


def require_training_space(directory, required_bytes):
    directory = Path(directory).expanduser()
    while not directory.exists():directory = directory.parent
    available = shutil.disk_usage(directory).free
    mount = json.loads(subprocess.check_output(['findmnt','--json','--output','TARGET,OPTIONS',
        '--target',str(directory)],text=True,timeout=10))['filesystems'][0]
    options = set(mount['options'].split(','))
    remaining = None
    if options.intersection({'quota','usrquota','uquota'}):
        result = subprocess.run(['quota','-w','-v','-f',mount['target']],
            text=True,capture_output=True,timeout=10)
        if result.returncode not in (0,1):raise RuntimeError('Could not read destination user quota')
        remaining = quota_remaining(result.stdout)
        if not re.search(r'^\s*\S+\s+\d+\*?\s+\d+\s+\d+',result.stdout,re.M):
            raise RuntimeError('Destination enforces user quota but no usable quota record was returned')
        if remaining is not None:available = min(available,remaining)
    needed = int(required_bytes*1.02)+1024**3
    if needed > available:
        raise OSError(28,f'Insufficient training destination capacity/quota: need {needed/1024**3:.2f} GiB, '
            f'available {available/1024**3:.2f} GiB at {directory}')
    return {'required_bytes':required_bytes,'available_bytes':available,
            'user_quota_remaining_bytes':remaining,'reserve_bytes':needed-required_bytes}
