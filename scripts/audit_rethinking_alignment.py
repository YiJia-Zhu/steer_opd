#!/usr/bin/env python3
"""Compare the project defaults with the pinned shell and inherited Hydra YAML.

The upstream shell is parsed as text, never executed. This audit performs no
model loading, Ray initialization, GPU admission or downloads.
"""

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml

from steer_opd.rethinking_protocol import (
    ROOT, build_overrides, compose_reference, load_profile, reference_commit,
)


REPORT_DIR = ROOT / "audit/rethinking_alignment_2026-11-12"


def literal(value):
    if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]*)?[eE][-+]?[0-9]+", value):
        return float(value)
    return yaml.safe_load(value)


def get_path(data, path):
    for name in path.split("."):
        if not isinstance(data, dict) or name not in data:
            raise KeyError(path)
        data = data[name]
    return data


def simple_integer(expression):
    node = ast.parse(expression, mode="eval").body

    def visit(value):
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return value.value
        if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Mult)):
            left, right = visit(value.left), visit(value.right)
            return left + right if isinstance(value.op, ast.Add) else left * right
        raise ValueError("Only constant addition/multiplication are supported")

    return visit(node)


def read_shell(path):
    variables = {}
    command_text = path.read_text().split("python3 -m verl.trainer.main_ppo", 1)[1]
    for line in path.read_text().splitlines():
        match = re.match(r"export ([A-Z_]+)=(.*?)(?:\s+#.*)?$", line.strip())
        if not match:
            continue
        name, value = match.groups()
        default = re.fullmatch(r'\$\{[A-Z_]+:-(.*?)\}', value)
        if default:
            value = default[1]
        if value.startswith('"${') and value.endswith('}"'):
            default = re.fullmatch(r'"\$\{[A-Z_]+:-(.*?)\}"', value)
            if default:
                value = default[1]
        if not any(character in value for character in "$`"):
            variables[name] = literal(value)
    variables["MAX_MODEL_LEN"] = variables["MAX_PROMPT_LENGTH"] + max(
        variables["MAX_RESP_LENGTH"], variables["MAX_VAL_RESP_LENGTH"])
    variables["PPO_MAX_TOKEN_LEN_PER_GPU"] = max(32768, 1024 + variables["MAX_RESP_LENGTH"])
    overrides = {}
    for line in command_text.splitlines():
        match = re.match(r"\s*(\+?[a-z][\w.]+)=(.*?)\s*\\?\s*$", line)
        if not match:
            continue
        key, value = match.groups()
        if "$" in value:
            value = re.sub(r'\$\{([A-Z_]+)\}|\$([A-Z_]+)',
                           lambda m: str(variables.get(m[1] or m[2], "$UNKNOWN")), value)
        if value.startswith("$((") and value.endswith("))"):
            value = str(simple_integer(value[3:-2]))
        if any(character in value for character in "$`"):
            continue
        try:
            overrides[key.lstrip("+")] = literal(value)
        except yaml.YAMLError:
            continue
    return overrides


def old_launcher_values():
    path = REPORT_DIR / "before/scripts/launch_pilot.py"
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text())
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "values" for t in node.targets):
            for key, value in zip(node.value.keys, node.value.values):
                try:
                    result[ast.literal_eval(key)] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    result[ast.literal_eval(key)] = ast.unparse(value)
    return result


def audit():
    profile = load_profile()
    commit = reference_commit(profile)
    if commit != profile["reference"]["commit"]:
        raise ValueError("Pinned reference commit changed")
    upstream = ROOT / profile["reference"]["root"]
    shell = read_shell(ROOT / profile["reference"]["shell"])
    inherited = compose_reference(profile, {})
    values = build_overrides(profile, REPORT_DIR / "audit_preview")
    actual = compose_reference(profile, values)
    comparisons = []
    for raw_key, configured in profile["paper_hydra_overrides"].items():
        key = raw_key.lstrip("+")
        if key in shell:
            expected, source = shell[key], "official on_policy_distillation.sh"
        else:
            expected, source = get_path(inherited, key), "official inherited Hydra YAML"
        observed = get_path(actual, key)
        comparisons.append({"key": key, "reference": expected, "project": observed,
                            "reference_source": source, "matches": expected == observed})
    generator = upstream / "scripts/val/eval/gen_vllm.py"
    constants = {}
    for node in ast.parse(generator.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    eval_comparisons = []
    for key, source in (("max_tokens", "MAX_TOKENS"), ("temperature", "TEMPERATURE"), ("top_p", "TOP_P")):
        expected = constants[source]
        eval_comparisons.append({"key": key, "reference": expected,
                                 "project": profile["evaluation"][key],
                                 "matches": profile["evaluation"][key] == expected})
    tree = ast.parse(generator.read_text())
    task_counts = [ast.literal_eval(value) for node in ast.walk(tree) if isinstance(node, ast.Dict)
                   for key, value in zip(node.keys, node.values)
                   if isinstance(key, ast.Constant) and key.value == "N"]
    assert task_counts and set(task_counts) == {16}
    eval_comparisons.append({"key": "n", "reference": 16, "project": profile["evaluation"]["n"],
                             "matches": profile["evaluation"]["n"] == 16})
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "stop_token":
            markers = ast.literal_eval(node.iter)
            eval_comparisons.append({"key": "stop_markers", "reference": markers,
                                     "project": profile["evaluation"]["stop_markers"],
                                     "matches": markers == profile["evaluation"]["stop_markers"]})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_defaults":
            for keyword in node.keywords:
                if keyword.arg == "enable_thinking":
                    enabled = ast.literal_eval(keyword.value)
                    eval_comparisons.append({"key": "enable_thinking", "reference": enabled,
                                             "project": profile["evaluation"]["enable_thinking"],
                                             "matches": enabled == profile["evaluation"]["enable_thinking"]})
    old = old_launcher_values()
    differences = [{"key": key.lstrip("+"), "before": old.get(key.lstrip("+"), "not explicitly set"),
                    "now": value} for key, value in values.items()
                   if old.get(key.lstrip("+"), object()) != value]
    version_paths = [profile["reference"]["shell"],
                     profile["reference"]["python_root"] + "/verl/workers/actor/dp_actor.py",
                     profile["reference"]["python_root"] + "/verl/trainer/ppo/core_algos.py",
                     str(generator.relative_to(ROOT)), "configs/rethinking_opd.json"]
    result = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "reference_commit": commit,
        "reference_worktree_changes": subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).splitlines(),
        "paper_default_checks": comparisons, "evaluation_checks": eval_comparisons,
        "unexpected_differences": [x for x in comparisons + eval_comparisons if not x["matches"]],
        "old_launcher_default_differences": differences,
        "intentional_scope_exceptions": profile["scope_exceptions"],
        "loss": {"family": profile["protocol"], "backend": "original fork actor.compute_distillation_reward and compute_policy_loss_vanilla",
                 "newer_vendor_forward_kl_backend_used": False,
                 "paper_vs_code": "Paper defines renormalized subset KL; fork uses detached student_p-weighted native log-ratio rewards and clipped PPO surrogate. Preserve the fork, not a claimed exact gradient identity."},
        "sources": [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()} for name in version_paths],
        "execution_authorized": False, "gpu_or_download_started": False,
        "remaining": ["Original fork dependencies are not GPU-validated with the installed newer runtime.",
                      "Controlled rollout and learning replay are not implemented in this backend; steering execution is blocked.",
                      "100-question repeated parquet, independent calibration/dev splits and EOS/tokenizer support checks remain pending."]}
    return result


def main():
    result = audit()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"paper_defaults": len(result["paper_default_checks"]),
                      "evaluation_defaults": len(result["evaluation_checks"]),
                      "unexpected_differences": result["unexpected_differences"],
                      "gpu_or_download_started": False}, ensure_ascii=False, indent=2))
    if result["unexpected_differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
