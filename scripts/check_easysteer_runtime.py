#!/usr/bin/env python3
"""Check pinned EasySteer files and actual CPU APIs; never load GPU models."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sysconfig

cuda=Path(sysconfig.get_path('purelib'))/'nvidia/cu13'
if (cuda/'bin/nvcc').is_file():os.environ['CUDA_HOME']=str(cuda)

import torch
import easysteer.capture as hs
from steer_opd.easysteer_static import build_spec, require_easysteer
from steer_opd.easysteer_plugin import ALGORITHM
from vllm.capture import SelectSpec
from vllm.model_hooks.steering.algorithms.registry import ALGORITHM_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def main():
    require_easysteer()
    site = Path(sysconfig.get_path('purelib'))
    checked = 0
    for entry in json.loads((ROOT/'ref/easysteer_runtime/manifest.json').read_text())['files']:
        rel = Path(entry['path']).relative_to('ref/easysteer_runtime')
        if rel.parts[0] == 'vllm':
            assert hashlib.sha256((site/rel).read_bytes()).hexdigest() == entry['sha256'], rel
            checked += 1
    assert callable(hs.capture)
    capture_parameters=inspect.signature(hs.capture).parameters
    assert {'per_prompt_selects','layers','dtype','budget_bytes'} <= set(capture_parameters)
    # Sampling options such as seed are forwarded to SamplingParams as kwargs.
    assert capture_parameters['generate_kwargs'].kind is inspect.Parameter.VAR_KEYWORD
    from vllm import LLM
    from vllm.entrypoints.llm import _resolve_steering
    assert {'steering','use_tqdm'} <= set(inspect.signature(LLM.generate).parameters)
    assert _resolve_steering(False) is False
    assert SelectSpec(prompt_positions=[2,4]).to_wire()['prompt_positions'] == [2,4]
    row = {'id':'q','prompt_token_ids':[1,2,3],'token_ids':[4,5,6]}
    vector = {26:torch.ones(2560)}
    assert build_spec(row,[True,False,True],vector) is not False
    assert build_spec(row,[False]*3,vector) is False
    assert build_spec(row,[True]*3,vector,alpha=0) is False
    algorithm = ALGORITHM_REGISTRY[ALGORITHM]()
    algorithm.set_payload(torch.ones(4),scale_factor=.1)
    hidden = torch.tensor([[1.,2.,3.,4.]])
    changed = algorithm._transform(hidden,torch.ones(4))
    assert torch.isfinite(changed).all() and not torch.equal(changed,hidden)
    assert torch.allclose(changed.norm(dim=-1),hidden.norm(dim=-1),atol=1e-5)
    checks = {'pinned_fork_files_match':True, 'pinned_files_checked':checked, 'capture_api':True,
        'selector':True,'static_slerp_authoring':True,'empty_and_alpha0_disable':True,'public_generate_api':True,
        'slerp_registry_and_norm':True,'gpu_started':False,
        'versions':{p:metadata.version(p) for p in ['vllm','easysteer','torch','transformers']}}
    output = ROOT/'audit/dual_overlap_implementation_20261114/runtime_cpu_check.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(checks,indent=2)+'\n')
    print(checks)


if __name__ == '__main__':
    main()
