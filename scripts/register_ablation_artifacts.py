#!/usr/bin/env python3
"""Register old committed overlap stages as SHA references. CPU only; no models run."""
import argparse
import json
import os

from steer_opd.legacy_stage_import import register
from steer_opd.resumable_core import read_json
from steer_opd.stage_io import configure_layout,project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ablation.json')
    parser.add_argument('--legacy-run',required=True)
    parser.add_argument('--param-name',help='Optional original-parameter display name')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    legacy=project(args.legacy_run)
    if (legacy.parent/'historical_run.json').exists():
        from steer_opd.artifact_reuse import StageArtifacts
        saved=read_json(legacy.parent/'layout_config.json');cache=StageArtifacts(saved)
        for ref in (legacy.parent/'upstreams').glob('*.json'):cache.validate(project(read_json(ref)['descriptor']))
        print('Already physically migrated; all registered artifact hashes validated.')
        return
    cfg=read_json(project(args.config));old=read_json(legacy/'config.json')
    for key in ('question_count','alpha','layers','teacher_gpus','gpu_memory_utilization','sv_max_context_len',
                'reference_max_tokens','source_max_tokens','reference_temperature','reference_top_p','select_fraction'):
        cfg[key]=old[key]
    cfg.update(input_data_dir=old['data_dir'],param_name=args.param_name,include_teacher_ablation=False)
    cfg=configure_layout(cfg)
    class Progress:
        def event(self,event,**fields):print(f"Registered: {fields.get('name',event)}",flush=True)
    print(json.dumps(register(cfg,args.legacy_run,Progress()),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
