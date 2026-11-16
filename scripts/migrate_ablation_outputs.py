#!/usr/bin/env python3
"""Move the completed 100/1000-question panels into outputs. No GPU/model calls."""
import argparse
import os
from steer_opd.output_migration import build_plan,execute,LEDGER
from steer_opd.resumable_core import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Move completed artifacts and delete the listed unused runs')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    plan=read_json(LEDGER) if LEDGER.exists() else build_plan()
    print({'files_to_move':len(plan['moves']),'unused_results_to_delete':plan['delete']},flush=True)
    if args.execute:execute(plan)


if __name__=='__main__':main()
