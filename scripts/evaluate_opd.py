#!/usr/bin/env python3
"""Evaluate a trained bare student; run under the idle-GPU supervisor."""
import argparse
import json

from steer_opd.opd_evaluation import run_evaluation
from steer_opd.resumable_core import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(read_json(args.job)), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
