"""Removal of runtime caps must not invalidate completed answers/checkpoints."""

import json

import pytest

from steer_opd.generation_runtime import (
    checkpoint_config_hash, default_scheduler_job, prepare_generation_manifest,
)
from steer_opd.resumable_core import atomic_json, digest


def legacy_job():
    return {"model": "frozen-model", "output": "answers", "temperature": 0.7, "top_p": 0.95,
            "max_tokens": 31744, "support_size": 151665, "vector": None, "max_num_seqs": 16,
            "requests": [{"key": "a", "prompt_token_ids": [1, 2], "seed": 11},
                         {"key": "b", "prompt_token_ids": [3, 4], "seed": 12}]}


def test_cap_removal_reuses_completed_answer_bytes_and_preserves_provenance(tmp_path):
    job = legacy_job()
    old_hash = digest(job)
    atomic_json(tmp_path / "job.json", {"hash": old_hash, "job": job})
    answer = {**job["requests"][0], "job_hash": old_hash, "token_ids": [7, 9], "text": "saved"}
    atomic_json(tmp_path / "a.json", answer)
    before = (tmp_path / "a.json").read_bytes()
    effective, new_hash, pending = prepare_generation_manifest(tmp_path, default_scheduler_job(job))
    assert [r["key"] for r in pending] == ["b"]
    assert "max_num_seqs" not in effective
    assert new_hash != old_hash
    assert (tmp_path / "a.json").read_bytes() == before
    assert json.loads((tmp_path / "job.json").read_text())["accepted_job_hashes"] == [old_hash, new_hash]
    assert (tmp_path / "job_history" / f"{old_hash}.json").exists()
    assert prepare_generation_manifest(tmp_path, effective)[2] == pending


@pytest.mark.parametrize("key,value", [("temperature", 1.0), ("max_tokens", 8192),
                                       ("support_size", 151669), ("vector", "new-vector")])
def test_runtime_migration_rejects_scientific_changes(tmp_path, key, value):
    job = legacy_job()
    prepare_generation_manifest(tmp_path, job)
    changed = {**job, key: value}
    before = (tmp_path / "job.json").read_bytes()
    with pytest.raises(ValueError, match="scientific generation"):
        prepare_generation_manifest(tmp_path, changed)
    assert (tmp_path / "job.json").read_bytes() == before


def test_completed_answer_with_wrong_seed_or_provenance_is_rejected(tmp_path):
    job, hash_, _ = prepare_generation_manifest(tmp_path, legacy_job())
    row = {**job["requests"][0], "job_hash": hash_, "seed": 99}
    atomic_json(tmp_path / "a.json", row)
    with pytest.raises(ValueError, match="request changed"):
        prepare_generation_manifest(tmp_path, job)
    row.update(seed=11, job_hash="unrecognized")
    atomic_json(tmp_path / "a.json", row)
    with pytest.raises(ValueError, match="provenance"):
        prepare_generation_manifest(tmp_path, job)


def test_deleting_legacy_config_caps_keeps_optimizer_identity(tmp_path):
    original = {"run_dir": str(tmp_path), "learning_rate": 1e-6,
                "generation_batch_size": 32, "vllm_max_num_seqs": 16}
    atomic_json(tmp_path / "config.json", original)
    current = {k: v for k, v in original.items() if k not in ["generation_batch_size", "vllm_max_num_seqs"]}
    assert checkpoint_config_hash(current) == digest(original)
    with pytest.raises(ValueError, match="frozen scientific"):
        checkpoint_config_hash({**current, "learning_rate": 1e-5})
