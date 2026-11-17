"""Independent exploration paths; all historical inputs are read-only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'exploration/snapshot/src'
sys.path.insert(0,str(SNAPSHOT))
PAIR='T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared'
NAME='st_direction_design_n1000_seed42_v3'
TS=ROOT/'outputs/step_3_overlap/ablation'/PAIR/'n1000_slerp0.1_layer26_seed42_v1'
ST=ROOT/'outputs/step_3_overlap/ablation'/PAIR/'n1000_slerp0.1_S26_T29_seed42_st_only_v1'
DATA=ROOT/'outputs/step_0_data/ablation'/PAIR/'n1000_slerp0.1_layer26_seed42_v1'
POOL=ROOT/'outputs/step_2_sv/ablation'/PAIR/'n1000_slerp0.1_S26_T29_seed42_st_only_v1/reader_teacher/activations/pooled.pt'
BANK29=POOL.parent.parent/'banks/S-T_global.pt'
BANK19=ROOT/'outputs/step_2_sv/param'/PAIR/'n1000_slerp0.05_S26_T19_seed42_st_all_global_v2/reader_teacher/banks/S-T_global.pt'
OUT=ROOT/'outputs/step_3_overlap/exploration'/PAIR/NAME
SUMMARY=ROOT/'outputs/step_6_summary/exploration'/PAIR/NAME
VECTORS=ROOT/'outputs/step_2_sv/exploration'/PAIR/NAME
from steer_opd.resumable_core import read_json,atomic_json,atomic_torch,digest
from steer_opd.direction_bank import file_sha256

def input_path(value):
 from steer_opd.compass_job import project_path
 return project_path(value)

def load_packet(path):
 import torch
 return torch.load(path,map_location='cpu',weights_only=True)
