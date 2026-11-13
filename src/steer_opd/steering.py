from contextlib import contextmanager, ExitStack

import torch

from steer_opd.operators import transform, validate_operator


def prediction_mask(response_mask):
    """A response token at t is predicted by the state at t-1.

    Shift each example separately so a packed/batch boundary never wraps.
    The last input state does not predict an observed response token.
    """
    if response_mask.ndim != 2:
        raise ValueError("Expected padded [batch, sequence] response mask")
    mask = torch.zeros_like(response_mask, dtype=torch.bool)
    mask[:, :-1] = response_mask[:, 1:].bool()
    return mask


def response_prediction_mask(prefix_length, response_length, gates=None, *, device=None):
    if prefix_length < 1 or response_length < 1:
        raise ValueError("Empty prompt/response")
    if gates is None:
        gates = [True] * response_length
    if len(gates) != response_length or any(type(g) is not bool for g in gates):
        raise ValueError("Recorded steering mask must contain one boolean per response token")
    result = torch.zeros((1, prefix_length + response_length - 1), dtype=torch.bool, device=device)
    result[:, prefix_length - 1:] = torch.tensor(gates, dtype=torch.bool, device=device)
    return result


class SteeringController:
    """One fixed vector/operator, enabled across forward AND backward."""

    def __init__(self, layer, vector, alpha, *, operator="additive"):
        validate_operator(operator, float(alpha))
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError("Steering vector must be a finite one-dimensional tensor")
        self.vector = vector.detach().clone()
        self.alpha = float(alpha)
        self.operator = operator
        self.nonzero = bool(torch.count_nonzero(self.vector))
        self.mask = None
        self.enabled = False
        self.injected_forwards = 0
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if self.mask is None or tuple(hidden.shape[:2]) != tuple(self.mask.shape):
            raise RuntimeError("Steering prediction mask does not match decoder states")
        if hidden.shape[-1] != self.vector.numel():
            raise RuntimeError("Steering vector width does not match the student")
        self.injected_forwards += 1
        if self.alpha == 0 or not self.nonzero:
            return output
        if self.operator == "additive":
            delta = self.vector.to(device=hidden.device, dtype=hidden.dtype) * self.alpha
            steered = hidden + self.mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1) * delta
        else:
            changed = transform(hidden, self.vector, self.alpha, self.operator)
            steered = torch.where(self.mask.to(device=hidden.device).unsqueeze(-1), changed, hidden)
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    @contextmanager
    def loss_scope(self, mask=None):
        if self.enabled:
            raise RuntimeError("Nested steering scopes are not supported")
        self.mask, self.enabled = mask, True
        try:
            yield self
        finally:
            self.mask, self.enabled = None, False

    def detach(self):
        if self.enabled:
            raise RuntimeError("Cannot detach while forward/backward is active")
        self.handle.remove()


class CompositeSteeringController:
    """Each block uses its own direction and scale; one shared prediction mask."""

    def __init__(self, model, vectors, *, operator="additive", alpha=1.0):
        self.controllers = [SteeringController(find_decoder_layer(model, layer), vector, alpha, operator=operator)
                            for layer, vector in vectors.items()]

    @property
    def enabled(self):
        return any(c.enabled for c in self.controllers)

    @contextmanager
    def loss_scope(self, mask=None):
        with ExitStack() as stack:
            for controller in self.controllers:
                stack.enter_context(controller.loss_scope(mask))
            yield self

    def detach(self):
        for controller in self.controllers:
            controller.detach()


def find_decoder_layer(model, layer_index):
    matches = [(name, layer) for name, layer in model.named_modules()
               if name.endswith(f"layers.{layer_index}")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one decoder block {layer_index}, found {[n for n, _ in matches]}")
    return matches[0][1]
