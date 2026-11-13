#!/usr/bin/env python3
"""Real-model HF cache/replay and vLLM/HF distribution acceptance."""

import argparse
from contextlib import nullcontext
import gc
import os

import torch
from transformers import AutoTokenizer

from steer_opd.resumable_core import (ROOT, MasterAdam, atomic_json, atomic_torch, controller_for,
                                      emit, load_model, read_json, response_hidden, response_loss, score_hidden)
from steer_opd.resumable_core import project_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_first_run.json")
    parser.add_argument("--phase", choices=["hf", "compare", "backward"], default="hf")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    cfg = read_json(args.config)
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    run = ROOT / cfg["run_dir"] / "acceptance"
    run.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    model = load_model(ROOT / cfg["student"])
    samples = read_json(ROOT / cfg["data_dir"] / "dev.json")[:3]
    if args.phase == "backward":
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reports = []
        for name in ["plain", "steer"]:
            model = load_model(ROOT / cfg["student"])
            hook = controller_for(model, run / "test_vector.pt" if name == "steer" else None)
            row = read_json(run / (name + args.suffix) / "sample_0.json")
            with torch.inference_mode():
                scores = score_hidden(model, response_hidden(model, row, hook))
            teacher = load_model(ROOT / cfg["teacher"])
            with torch.inference_mode():
                t = score_hidden(teacher, response_hidden(teacher, row), candidates=scores["ids"])
            scores["teacher_logps"] = t["logps"]
            del teacher, t
            gc.collect()
            torch.cuda.empty_cache()
            model.requires_grad_(True).train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer = MasterAdam(model, cfg)
            before = optimizer.master[-1].detach().clone()
            loss, error = response_loss(model, row, scores, hook, len(row["token_ids"]))
            optimizer.accumulate()
            norm = optimizer.step(cfg["grad_clip"])
            change = float((optimizer.master[-1].detach() - before).abs().max())
            assert norm > 0 and change > 0 and error < 0.2, (norm, change, error)
            reports.append({"policy": name, "loss": loss, "grad_norm": norm,
                            "master_parameter_max_change": change, "score_replay_max_logp_error": error})
            if hook:
                assert not hook.enabled
                hook.detach()
            del optimizer, model, hook, before, scores
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(run / "backward.json", {"passed": True, "updates": reports,
                                             "experimental_updates": 0})
        emit("real_model_backward_passed", updates=reports)
        return
    if args.phase == "hf":
        delta = torch.linspace(-0.03, 0.03, model.config.hidden_size)
        atomic_torch(run / "test_vector.pt", {"delta": delta, "layer": cfg["layer"]})
        hook = controller_for(model, run / "test_vector.pt")
        prefix = samples[0]["student_prompt_ids"]
        suffix = tokenizer.encode("The result follows from the given equation. We obtain 42.", add_special_tokens=False)
        ids = torch.tensor([prefix + suffix], device="cuda")
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, len(prefix) - 1:] = True
        with torch.inference_mode():
            baseline = model(ids, use_cache=False).logits[:, len(prefix) - 1:].float()
            hook.alpha = 0.0
            with hook.loss_scope(mask):
                zero = model(ids, use_cache=False).logits[:, len(prefix) - 1:].float()
            assert torch.equal(baseline, zero)
            hook.alpha = 1.0
            with hook.loss_scope(mask):
                full = model(ids, use_cache=False).logits[:, len(prefix) - 1:].float()
            cache, parts = None, []
            for end in range(len(prefix), ids.shape[1] + 1):
                tokens = ids[:, :end] if cache is None else ids[:, end - 1:end]
                current_mask = torch.zeros_like(tokens, dtype=torch.bool)
                current_mask[:, -1] = True
                with hook.loss_scope(current_mask):
                    output = model(tokens, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                parts.append(output.logits[:, -1].float())
            cached = torch.stack(parts, 1)
            lp, cp = full.log_softmax(-1), cached.log_softmax(-1)
            kl = (lp.exp() * (lp - cp)).sum(-1).mean().item()
            assert kl < 0.005, kl
            assert not torch.equal(baseline, full)
            restored = model(ids, use_cache=False).logits[:, len(prefix) - 1:].float()
            assert torch.equal(restored, baseline)
        hook.detach()
        atomic_json(run / "hf.json", {"passed": True, "alpha_zero_exact": True,
                                        "hook_removal_exact": True, "cache_replay_mean_kl": kl,
                                        "threshold_mean_kl": 0.005})
        for name in ["plain", "steer"]:
            requests = [{"key": f"sample_{i}", "id": r["id"], "prompt_token_ids": r["student_prompt_ids"],
                         "seed": 42 + i} for i, r in enumerate(samples)]
            atomic_json(run / f"job_{name}.json", {
                "model": cfg["student"], "output": str((run / name).relative_to(ROOT)),
                "requests": requests, "temperature": 0.0, "top_p": 1.0, "max_tokens": 32,
                "batch_size": 3, "max_num_seqs": 3, "gpu_memory_utilization": 0.6,
                "logprobs": 20, "vector": str((run / "test_vector.pt").relative_to(ROOT)) if name == "steer" else None,
                "support_size": cfg["shared_vocabulary_size"],
            })
        emit("hf_acceptance_passed", cache_replay_mean_kl=kl)
    else:
        report = []
        for name in ["plain", "steer"]:
            hook = controller_for(model, run / "test_vector.pt" if name == "steer" else None)
            for i in range(3):
                row = read_json(run / (name + args.suffix) / f"sample_{i}.json")
                with torch.inference_mode():
                    hidden = response_hidden(model, row, hook)
                    lp = project_logits(model, hidden).log_softmax(-1)
                errors, greedy, full_kls, sampled_errors = [], [], [], []
                for t, dictionary in enumerate(row["top_logprobs"]):
                    for token, value in dictionary.items():
                        errors.append(abs(float(lp[t, int(token)]) - value))
                    greedy.append(int(lp[t].argmax()) == row["token_ids"][t])
                    sampled_errors.append(abs(float(lp[t, row["token_ids"][t]]) - row["sampled_logps"][t]))
                    capture_dir = run / (name + args.suffix) / "full_logits"
                    capture = capture_dir / f"{row.get('capture_request_id', row.get('vllm_request_id'))}_{t}.pt"
                    if not capture.exists():
                        legacy = list(capture_dir.glob(f"{row.get('vllm_request_id')}-*_{t}.pt"))
                        if len(legacy) == 1:
                            capture = legacy[0]
                    if capture.exists():
                        reference = torch.load(capture, weights_only=True).to(lp.device).log_softmax(-1)
                        full_kls.append(float((reference.exp() * (reference - lp[t])).sum()))
                mean, maximum = sum(errors) / len(errors), max(errors)
                report.append({"policy": name, "sample": i, "mean_logp_error": mean,
                               "max_logp_error": maximum, "greedy_agreement": sum(greedy) / len(greedy),
                               "sampled_mean_logp_error": sum(sampled_errors) / len(sampled_errors),
                               "full_vocab_kls": full_kls})
            if hook:
                hook.detach()
        # Same full-distribution KL tolerance as the HF cached/full replay test.
        # Unweighted maxima over low-probability candidates are diagnostic,
        # rather than a stand-in for distributional agreement.
        passed = all(r["full_vocab_kls"] and sum(r["full_vocab_kls"]) / len(r["full_vocab_kls"]) < 0.005
                     and max(r["full_vocab_kls"]) < 0.03 and r["sampled_mean_logp_error"] < 0.06 for r in report)
        atomic_json(run / f"backend_comparison{args.suffix}.json", {"passed": passed, "comparisons": report,
                                                        "thresholds": {"mean_full_kl": 0.005, "max_full_kl": 0.03,
                                                                       "sampled_mean_logp_error": 0.06}})
        assert passed, report
        emit("vllm_hf_acceptance_passed", comparisons=report)


if __name__ == "__main__":
    main()
