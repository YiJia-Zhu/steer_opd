#!/usr/bin/env bash
set -euo pipefail

publication_mode="${1:---check}"
case "$publication_mode" in
    --check|--push) ;;
    *)
        printf '%s\n' '用法 bash publish_both.sh --check 或 --push' >&2
        exit 2
        ;;
esac

project_root=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
cd -- "$project_root"
runner_remote=git@github.com:YiJia-Zhu/steer-opd-runner.git
case "$(git remote get-url --push origin)" in
    git@github.com:YiJia-Zhu/steer_opd.git|https://github.com/YiJia-Zhu/steer_opd.git) ;;
    *)
        printf '%s\n' 'origin 需要指向 YiJia-Zhu/steer_opd，请核对远端地址。' >&2
        exit 1
        ;;
esac

research_commit=$(git rev-parse --verify refs/heads/main)
printf '完整版 main %.12s\n' "$research_commit"

if [[ -z "${GIT_SSH_COMMAND:-}" ]]; then
    publisher_ssh_command=$(git config --get core.sshCommand || true)
    if [[ -n "$publisher_ssh_command" ]]; then
        export GIT_SSH_COMMAND="$publisher_ssh_command"
    fi
fi

# Start from the remote runner history and remove retired files on updates.
mkdir -p -- "$project_root/.runtime"
publication_work=$(mktemp -d "$project_root/.runtime/runner-update.XXXXXXXX")
trap 'rm -rf -- "$publication_work"' EXIT
runner_checkout="$publication_work/runner"
runner_clone_url="$runner_remote"
if [[ "$publication_mode" == --check ]]; then
    runner_clone_url=https://github.com/YiJia-Zhu/steer-opd-runner.git
fi
git clone --quiet --single-branch --branch main "$runner_clone_url" "$runner_checkout"
git -C "$runner_checkout" rm -r -q -- .
git archive "$research_commit" -- \
    .gitignore src scripts runtime configs assets/metadata/Qwen3.5-35B-A3B \
    ref/easysteer_runtime/manifest.json ref/implementation_sources/rethinking_comparison.json |
    tar -x -C "$runner_checkout" \
        --exclude='scripts/migrate_to_186.sh' \
        --exclude='scripts/backup_static_sv.sh' \
        --exclude='scripts/migrate_ablation_outputs.py' \
        --exclude='scripts/register_ablation_artifacts.py' \
        --exclude='scripts/validate_package.py'

git -C "$runner_checkout" add -A
python3 - "$runner_checkout" <<'PY'
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1])
listing = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
sensitive_suffixes = {'.md', '.pdf', '.pem', '.key', '.log', '.pt', '.safetensors'}
credential_pattern = re.compile(
    rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----'
    rb'|gh[pours]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}'
)
paths = [root / item.decode() for item in listing.split(b'\0') if item]
for path in paths:
    blocked = path.is_symlink() or path.suffix.lower() in sensitive_suffixes or path.name.startswith('.env')
    if blocked or credential_pattern.search(path.read_bytes()):
        raise SystemExit(f'公开版含需要审查的文件 {path.relative_to(root)}')
print(f'公开版预览 {len(paths)} 个文件')
PY

if git -C "$runner_checkout" diff --cached --quiet; then
    printf '%s\n' '公开版文件无变更。'
else
    git -C "$runner_checkout" diff --cached --stat
    if [[ "$publication_mode" == --push ]]; then
        git -C "$runner_checkout" \
            -c user.name="$(git config user.name)" \
            -c user.email="$(git config user.email)" \
            commit -m 'Update runner'
    fi
fi

if [[ "$publication_mode" == --push ]]; then
    git push origin "$research_commit:refs/heads/main"
    git -C "$runner_checkout" push origin HEAD:refs/heads/main
fi
