from contextlib import contextmanager

import torch


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


class SteeringController:
    """One fixed additive vector, enabled explicitly across forward AND backward."""

    def __init__(self, layer, vector, alpha):
        self.vector = vector.detach().clone()
        self.alpha = float(alpha)
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
        if self.alpha == 0:
            return output
        delta = self.vector.to(device=hidden.device, dtype=hidden.dtype) * self.alpha
        steered = hidden + self.mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1) * delta
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


def find_decoder_layer(model, layer_index):
    matches = [(name, layer) for name, layer in model.named_modules()
               if name.endswith(f"layers.{layer_index}")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one decoder block {layer_index}, found {[n for n, _ in matches]}")
    return matches[0][1]
