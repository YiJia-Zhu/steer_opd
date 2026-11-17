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
        from steer_opd.easysteer_static import crop_native_support
        crop_native_support(self.model_runner.get_model(), support)
