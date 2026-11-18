"""Explicitly approved training storage, separate from static experiment outputs."""
from pathlib import Path

from steer_opd.resumable_core import ROOT

DEFAULT_TRAIN_ROOT = '~/steer_opd_outputs'


def training_path(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT/path).resolve()
    approved = (Path.home()/'steer_opd_outputs').resolve()
    if not (path.is_relative_to(ROOT.resolve()) or path.is_relative_to(approved)):
        raise ValueError('Training files must stay in the project or ~/steer_opd_outputs')
    return path


def stored_path(value):
    """Keep historical project-relative identities; external paths are absolute."""
    path = training_path(value)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def training_output_dir(experiment, run_name, pair):
    from steer_opd.stage_io import safe_name
    root = training_path(experiment.get('train_output_root', DEFAULT_TRAIN_ROOT))
    return root/'step_4_train/ablation'/pair/safe_name(run_name)


def is_retirable_training_checkpoint(path):
    path = training_path(path)
    roots = (ROOT/'outputs/step_4_train', Path.home()/'steer_opd_outputs/step_4_train')
    return any(path.is_relative_to(root.resolve()) for root in roots)
