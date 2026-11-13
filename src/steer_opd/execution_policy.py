"""Explicit experiment holds and default vLLM scheduler requirements."""


POLICY_KEYS = ("execution_hold", "execution_blockers")
SEQUENCE_LIMIT_OPTIONS = frozenset({
    "max_num_seqs", "max_num_active_seqs", "vllm_max_num_seqs",
    "num_seqs", "num_seq", "seq_num", "max-num-seqs",
    "generation_batch_size", "cudagraph_capture_sizes", "max_cudagraph_capture_size",
})


def execution_policy(config):
    """Carry the user's hold into every separately executable stage job."""
    return {key: config[key] for key in POLICY_KEYS if key in config}


def require_execution_allowed(config):
    """Call before model/GPU admission, launching subprocesses or cache writes."""
    if config.get("execution_hold"):
        raise RuntimeError(f"Experiment execution is held: {config['execution_hold']}")
    if config.get("execution_blockers"):
        raise RuntimeError(f"Experiment prerequisites remain: {config['execution_blockers']}")


def require_default_scheduler_options(config):
    """Reject new manual sequence/graph caps; HF scoring batch sizes are separate."""
    def inspect(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in SEQUENCE_LIMIT_OPTIONS:
                    raise ValueError(f"Leave vLLM sequence scheduling at its defaults: {path}{key}")
                inspect(child, f"{path}{key}.")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}{index}.")
    inspect(config, "")
