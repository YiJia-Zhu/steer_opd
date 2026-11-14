"""Worker support cropping for an isolated EasySteer-fork environment."""

import os

from steer_opd.easysteer_plugin import register

if not register():
    raise ImportError("StaticWorker requires the supported EasySteer model_hooks fork")

from vllm.v1.worker.gpu_worker import Worker


class StaticWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        support = int(os.environ["STEER_OPD_STATIC_SUPPORT"])
        from steer_opd.engine_projection import crop_native_support
        crop_native_support(self.model_runner.get_model(), support)
        projection = os.environ.get("STEER_OPD_BYTE_PROJECTION")
        self._project_projection = None
        if projection:
            import torch
            from steer_opd.engine_projection import install_projection
            self._project_projection = install_projection(self.model_runner,
                torch.load(projection, map_location="cpu", weights_only=True),
                capture_sampled=os.environ.get("STEER_OPD_CAPTURE_SAMPLED_BYTES", "0") == "1")

    def project_pop_byte_statistics(self, request_id):
        return self._project_projection.pop(request_id) if self._project_projection else None
