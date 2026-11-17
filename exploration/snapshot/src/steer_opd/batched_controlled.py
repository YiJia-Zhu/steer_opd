"""Per-request tensor hooks for the explicit HF fixed-prefix reference."""
import torch


class BatchedController:
    def __init__(self, model, vectors, *, operator="additive", alpha=1.0, record_perturbations=False):
        from steer_opd.operators import validate_operator
        strengths=list(alpha) if isinstance(alpha,(list,tuple)) else None
        for value in strengths if strengths is not None else [alpha]:validate_operator(operator,value)
        if not vectors or any(value.ndim != 2 or not torch.isfinite(value).all() for value in vectors.values()):
            raise ValueError("Batched vectors must be finite [batch,hidden] tensors")
        self.mask = None
        self.indices = None
        self.handles = []
        self.perturbations = {}
        self.nonzero_rows = torch.stack([value.float().norm(dim=-1)>0 for value in vectors.values()]).any(0).tolist()
        if strengths is not None and len(strengths)!=len(self.nonzero_rows):
            raise ValueError('One reference strength per request is required')
        for layer, delta in vectors.items():
            def hook(module, args, output, delta=delta, layer_index=layer):
                if self.mask is None:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                values = delta if self.indices is None else delta[self.indices]
                if hidden.shape[:2] != self.mask.shape or hidden.shape[0] != values.shape[0]:
                    raise ValueError("Batched steering shape mismatch")
                per_row=strengths if self.indices is None or strengths is None else [strengths[int(i)] for i in self.indices]
                if operator == "additive":
                    delta = values.to(hidden)[:, None, :]
                    factor=torch.tensor(per_row,device=hidden.device,dtype=hidden.dtype)[:,None,None] if per_row is not None else alpha
                    delta = delta * factor
                    changed = hidden + self.mask.to(hidden.dtype)[..., None] * delta
                else:
                    from steer_opd.operators import slerp
                    # Explicit HF reference only; one model forward retains heterogeneous strengths.
                    transformed=(slerp(hidden,values[:,None,:],alpha) if per_row is None else
                        torch.cat([slerp(hidden[i:i+1],values[i:i+1,None,:],a) for i,a in enumerate(per_row)]))
                    changed = torch.where(self.mask[..., None], transformed, hidden)
                if record_perturbations:
                    self.perturbations[layer_index] = ((changed.float()-hidden.float()).norm(dim=-1) /
                        hidden.float().norm(dim=-1).clamp_min(1e-8)).detach()
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            self.handles.append(model.model.layers[layer].register_forward_hook(hook))

    def scope(self, mask, indices=None):
        from contextlib import contextmanager
        @contextmanager
        def active():
            if self.mask is not None:
                raise ValueError("Nested batched intervention")
            self.mask, self.indices = mask, indices
            try:
                yield
            finally:
                self.mask, self.indices = None, None
        return active()

    def detach(self):
        for handle in self.handles:
            handle.remove()
