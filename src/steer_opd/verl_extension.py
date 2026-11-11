"""Opt-in extension for the pinned verl FSDP engine.

Enabled via actor_rollout_ref.model.external_lib=steer_opd.verl_extension.
No vLLM or teacher model is patched. Padded, SP=1, one actor minibatch/update
and no resume are required for this first implementation.
"""

import json
import os
from pathlib import Path

import torch

from steer_opd.steering import SteeringController, find_decoder_layer, prediction_mask
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead


def install():
    if getattr(FSDPEngine, "_steer_opd_installed", False):
        return
    original_batch = FSDPEngine.forward_backward_batch
    original_inputs = FSDPEngineWithLMHead.prepare_model_inputs

    def prepare_inputs(self, micro_batch):
        inputs, output_args = original_inputs(self, micro_batch)
        controller = getattr(self, "_steer_opd_controller", None)
        if controller is not None and controller.enabled:
            if tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True):
                raise RuntimeError("First steering implementation requires use_remove_padding=False")
            if self.ulysses_sequence_parallel_size != 1:
                raise RuntimeError("First steering implementation requires sequence parallelism=1")
            mask = micro_batch["loss_mask"]
            if mask.is_nested:
                mask = torch.nested.to_padded_tensor(mask, padding=0, output_size=inputs["input_ids"].shape)
            controller.mask = prediction_mask(mask)
            if not controller.mask.any():
                raise RuntimeError("An active steering batch has no response prediction positions")
        return inputs, output_args

    def forward_backward(self, data, loss_function, forward_only=False):
        config_file = os.environ.get("STEER_OPD_CONFIG")
        if not config_file or forward_only:
            return original_batch(self, data, loss_function, forward_only=forward_only)
        if not hasattr(self, "_steer_opd_settings"):
            self._steer_opd_settings = json.loads(Path(config_file).read_text())
            self._steer_opd_updates = 0
        config = self._steer_opd_settings
        active = self._steer_opd_updates < config["auxiliary_updates"]
        if active:
            if not hasattr(self, "_steer_opd_controller"):
                payload = torch.load(config["vector_path"], map_location="cpu", weights_only=True)
                self._steer_opd_controller = SteeringController(
                    find_decoder_layer(self.module, config["layer"]),
                    payload[config.get("vector_key", "direction")], config["alpha"],
                )
            controller = self._steer_opd_controller
            before = controller.injected_forwards
            with controller.loss_scope():
                result = original_batch(self, data, loss_function, forward_only=False)
            if controller.injected_forwards <= before:
                raise RuntimeError("Active steering did not reach the student's decoder")
            print(json.dumps({"steer_opd_update": self._steer_opd_updates + 1,
                              "injected_forward_calls": controller.injected_forwards - before,
                              "hook_disabled_after_backward": not controller.enabled}), flush=True)
        else:
            result = original_batch(self, data, loss_function, forward_only=False)
        self._steer_opd_updates += 1
        if self._steer_opd_updates == config["auxiliary_updates"]:
            self._steer_opd_controller.detach()
            print("STEER_OPD: auxiliary phase complete; student hook removed", flush=True)
        return result

    FSDPEngineWithLMHead.prepare_model_inputs = prepare_inputs
    FSDPEngine.forward_backward_batch = forward_backward
    FSDPEngine._steer_opd_installed = True


install()
