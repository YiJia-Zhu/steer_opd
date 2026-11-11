"""Check the runtime before each phase so dependency drift is explicit."""

import importlib.metadata
import sys


EXPECTED = {
    "torch": "2.13.0+cu130",
    "transformers": "5.16.1",
    "vllm": "0.32.0",
    "math-verify": "0.9.0",
    "verl": "0.10.0.dev0",
}


def verify_runtime():
    versions = {}
    for package in EXPECTED:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    mismatches = {name: {"expected": value, "actual": versions[name]}
                  for name, value in EXPECTED.items() if versions[name] != value}
    if mismatches:
        raise RuntimeError(
            f"OPD runtime differs from the validated baseline: {mismatches}. "
            "Use the project .venv-opd interpreter after sourcing scripts/runtime_env.sh."
        )
    return {"python": sys.version, "prefix": sys.prefix, "packages": versions}
