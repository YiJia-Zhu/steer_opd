"""Raw S-T/global additive vectors: apply the coefficient once before EasySteer."""
import torch

from steer_opd.operators import validate_operator
from steer_opd.resumable_core import digest


def raw_global_vectors(bank,layers,strength):
    validate_operator('additive',strength)
    if bank['method']['direction']!='S-T' or bank['method']['granularity']!='global':
        raise ValueError('Raw additive requires S-T/global bank')
    if not layers or len(set(layers))!=len(layers):raise ValueError('Select unique bank layers')
    vectors={}
    for layer in layers:
        direction=bank['layers'][str(layer)]['direction'].float()
        if not torch.isfinite(direction).all():raise ValueError('Nonfinite raw global difference')
        vectors[layer]=direction*strength
    identity={'bank_id':bank['bank_id'],'assigned':'global','operator':'additive',
              'strength':strength,'selected_layers':list(layers),
              'additive_scaling':'raw_mean_difference_v1','seed':bank['method']['seed']}
    return vectors,digest(identity)
