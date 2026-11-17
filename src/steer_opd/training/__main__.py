"""Read-only training-loss preview. No model, GPU, optimizer or output writes."""
import argparse
from dataclasses import replace
import json

from steer_opd.resumable_core import ROOT, read_json
from steer_opd.training import TrainingLossConfig, training_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ablation.json')
    parser.add_argument('--loss', choices=['traditional_opd', 'compass_opd'])
    parser.add_argument('--anchor-coef', type=float)
    parser.add_argument('--reference-model')
    args = parser.parse_args()
    experiment = read_json(ROOT / args.config)
    settings = TrainingLossConfig.from_experiment(experiment)
    changes = {key: value for key, value in {
        'loss_type': args.loss, 'anchor_coef': args.anchor_coef,
        'reference_model': args.reference_model}.items() if value is not None}
    print(json.dumps(training_plan(experiment, replace(settings, **changes)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
