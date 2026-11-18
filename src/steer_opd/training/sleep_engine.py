"""Sleep-enabled EasySteer factory for the two-card training adapter only."""
import os
from pathlib import Path
import sys


def create_sleep_engine(model_path,support_size,*,max_model_len,tensor_parallel_size,memory,batch_invariant=False,enforce_eager=False,disable_autotune=False):
    import torch
    from steer_opd.easysteer_static import require_easysteer_gpu,engine_options
    from steer_opd.experiment_runtime import compute_required,engine_startup_guard,cuda_toolkit
    from steer_opd.resumable_core import read_json
    compute_required();require_easysteer_gpu(max_gpus=tensor_parallel_size)
    if torch.cuda.device_count()!=tensor_parallel_size:raise ValueError('Scorer visibility must match admitted TP')
    cuda=cuda_toolkit(sys.prefix)
    os.environ['CUDA_HOME']=str(cuda)
    os.environ['PATH']=os.pathsep.join([str(Path(sys.prefix)/'bin'),str(cuda/'bin'),os.environ.get('PATH','')])
    os.environ['STEER_OPD_STATIC_SUPPORT']=str(support_size)
    options=engine_options(batch_invariant=batch_invariant)
    options['enable_prefix_caching']=False
    if tensor_parallel_size>1 or disable_autotune:options['enable_flashinfer_autotune']=False
    if 'vision_config' in read_json(Path(model_path)/'config.json'):
        options['limit_mm_per_prompt']={'image':0,'video':0}
    from vllm import LLM
    with engine_startup_guard():
        engine=LLM(model=str(model_path),dtype='bfloat16',tensor_parallel_size=tensor_parallel_size,
            seed=42,hf_overrides={'head_dtype':'float32'},max_model_len=max_model_len,
            gpu_memory_utilization=memory,enable_sleep_mode=True,enforce_eager=enforce_eager,logprobs_mode='raw_logprobs',
            max_logprobs=16,generation_config='vllm',worker_cls='steer_opd.easysteer_worker.StaticWorker',**options)
    return engine


class FrozenEngineResidency:
    """Acknowledge state only after the public blocking sleep/wake call returns."""
    def __init__(self,engine):self.engine=engine;self.state='awake'

    def sleep(self):
        if self.state=='asleep':return
        self.state='transitioning'
        self.engine.sleep(level=1,mode='wait')
        if not self.engine.llm_engine.is_sleeping():raise RuntimeError('Frozen engine did not acknowledge sleep')
        self.state='asleep'

    def wake(self):
        if self.state=='awake':return
        if self.state!='asleep':raise RuntimeError('Frozen engine has no confirmed sleeping state')
        self.state='transitioning'
        self.engine.wake_up()
        if self.engine.llm_engine.is_sleeping():raise RuntimeError('Frozen engine did not acknowledge wake')
        self.state='awake'
