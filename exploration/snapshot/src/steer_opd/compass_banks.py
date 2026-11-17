"""Two T-S banks from one frozen-student teacher/self activation pool."""
import torch

from steer_opd.direction_bank import make_bank
from steer_opd.method_config import DIRECTION_VERSIONS, Method


def make_common_banks(pooled, layers, identity, *, alpha=.1, direction='T-S'):
    valid={qid:record for qid,record in pooled.items() if record['usable']}
    if not valid:
        raise ValueError('No usable teacher/self activation pairs')
    for record in valid.values():
        for layer in layers:
            if len(record['layers'][str(layer)]['pools'])!=2:
                raise ValueError('T-S requires exactly student-read teacher/self activation pairs')
    scales={str(layer):torch.cat([norm.float() for record in valid.values()
            for norm in record['layers'][str(layer)]['norms']]).median() for layer in layers}
    result={}
    for granularity in ('global','per_question'):
        method=Method(direction=direction,granularity=granularity,extractor='initial',layers=tuple(layers),
                      gate='all',operator='slerp',alpha=alpha)
        result[f'{direction}_{granularity}']=make_bank(pooled,method,
            {**identity,'direction_version':DIRECTION_VERSIONS[direction],'zero_direction_policy':'exact_no_intervention'},
            fixed_scales=scales,allow_zero_directions=True)
    return result,scales
