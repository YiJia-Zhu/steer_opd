"""Default vLLM scheduling and auditable resume across runtime-only changes."""

from pathlib import Path

from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json


# Historical metadata only. These keys are never forwarded to an engine.
REMOVED_SEQUENCE_OPTIONS = frozenset({
    "batch_size", "max_num_seqs", "max_num_active_seqs",
    "cudagraph_capture_sizes", "max_cudagraph_capture_size",
})
REMOVED_CONFIG_OPTIONS = frozenset({"generation_batch_size", "vllm_max_num_seqs"})
GENERATION_RUNTIME_OPTIONS = REMOVED_SEQUENCE_OPTIONS | {
    "gpu_memory_utilization", "logprobs", "backend_version", "replay_audit_tokens", "scheduler",
}


def default_scheduler_job(job):
    result = {key: value for key, value in job.items() if key not in REMOVED_SEQUENCE_OPTIONS}
    result["scheduler"] = "vllm_defaults"
    return result


def scientific_generation_job(job):
    return {key: value for key, value in job.items() if key not in GENERATION_RUNTIME_OPTIONS}


def checkpoint_config_hash(cfg):
    """Retain old checkpoint identity after deleting unused runtime caps.

    The original frozen config is archival evidence. Every scientific setting
    must still match it, including data, sampling, steering, loss and eval budget.
    """
    frozen = ROOT / cfg["run_dir"] / "config.json"
    if not frozen.exists():
        return digest(cfg)
    original = read_json(frozen)
    clean = lambda value: {k: v for k, v in value.items() if k not in REMOVED_CONFIG_OPTIONS}
    if clean(original) != clean(cfg):
        raise ValueError("Refuse to change a frozen scientific experiment configuration")
    return digest(original)


def prepare_generation_manifest(output, job):
    """Keep completed response bytes and their original runtime provenance."""
    output = Path(output)
    path = output / "job.json"
    job = default_scheduler_job(job)
    current_hash = digest(job)
    accepted = [current_hash]
    if path.exists():
        previous = read_json(path)
        if previous["hash"] != digest(previous["job"]):
            raise ValueError("Corrupt generation job manifest")
        if scientific_generation_job(previous["job"]) != scientific_generation_job(job):
            raise ValueError("Cannot resume changed scientific generation settings")
        accepted = list(dict.fromkeys([*previous.get("accepted_job_hashes", []), previous["hash"], current_hash]))
        if previous["hash"] != current_hash:
            atomic_json(output / "job_history" / f"{previous['hash']}.json", previous)
    atomic_json(path, {"hash": current_hash, "job": job, "accepted_job_hashes": accepted})
    pending = []
    for request in job["requests"]:
        response_path = output / f"{request['key']}.json"
        if not response_path.exists():
            pending.append(request)
            continue
        response = read_json(response_path)
        if response.get("job_hash") not in accepted:
            raise ValueError(f"Unrecognized response provenance: {request['key']}")
        if any(response.get(key) != value for key, value in request.items()):
            raise ValueError(f"Completed response request changed: {request['key']}")
    return job, current_hash, pending
