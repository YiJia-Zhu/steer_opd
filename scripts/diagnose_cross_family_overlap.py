#!/usr/bin/env python3
"""Collect ordinary student states, then score all of them with one teacher.

No parameter updates. Original sampled IDs are retained for student scoring;
the teacher encodes the same visible answer prefix using its native tokenizer.
Every teacher re-encoding must round-trip to the exact shared text.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer_opd.cross_family_overlap import compare_distributions, visible_continuation
from steer_opd.math_data import TASK_SUFFIX, read_rows
from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_model(spec):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = ROOT / path
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cuda:0"},
    ).eval().requires_grad_(False)
    return tokenizer, model


def build_prompt(tokenizer, question, spec):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}], tokenize=True,
        return_dict=False, add_generation_prompt=True, **spec.get("chat_template_kwargs", {}),
    )


def describe_model(tokenizer, model, spec):
    mapping = tokenizer.get_vocab()
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"spec": spec, "tokenizer_mapping_sha256": digest, "mapped_tokens": len(mapping),
            "model_type": model.config.model_type, "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "native_eos_token_id": tokenizer.eos_token_id}


def verify_last_logit(model, input_ids):
    tokens = torch.tensor([input_ids], device="cuda")
    with torch.inference_mode():
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1:, :]
        direct = model.lm_head(hidden).float()
        native = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits.float()
    difference = (direct - native).abs().max().item()
    if difference > 1e-4:
        raise ValueError(f"Direct final-state scoring differs from native logits: {difference}")
    return {"native_logits_max_abs_difference": difference, "checked_context_tokens": len(input_ids)}


def score_state(tokenizer, model, input_ids, model_info, topk):
    tokens = torch.tensor([input_ids], device="cuda")
    with torch.inference_mode():
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden).float()
        probabilities = logits.softmax(-1)
        values, ids = probabilities[0].topk(topk)
    ids, values = ids.cpu().tolist(), values.cpu().tolist()
    prefix = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    special = set(tokenizer.all_special_ids)
    valid_ids = set(tokenizer.get_vocab().values())
    decodable = [index for index in ids if index in valid_ids and index not in special]
    decoded = tokenizer.batch_decode(
        [input_ids + [index] for index in decodable], skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded = dict(zip(decodable, decoded))
    rows = []
    for index, probability in zip(ids, values):
        row = {"token_id": index, "probability": probability,
               "continuation_hex": None, "byte_event": None}
        text = decoded.get(index, "")
        if index in eos:
            row.update(continuation_hex="EOS", byte_event="256")
        elif index in valid_ids and index not in special and text.startswith(prefix):
            suffix = text[len(prefix):]
            if suffix and "\ufffd" not in suffix:
                value = suffix.encode("utf-8")
                row.update(continuation_hex=value.hex(), byte_event=str(value[0]))
        rows.append(row)
    return {"tokenizer_mapping_sha256": model_info["tokenizer_mapping_sha256"],
            "native_context_tokens": len(input_ids), "top_tokens": rows,
            "native_topk_mass": sum(values)}


def student_phase(config, spec, out):
    tokenizer, model = load_model(spec)
    info = describe_model(tokenizer, model, spec)
    rows = read_rows(ROOT / config["dev_path"])[:config["question_limit"]]
    cached = None
    if config.get("reuse_rollouts_from"):
        cached = {r["id"]: r for r in read_rows(ROOT / config["reuse_rollouts_from"] / spec["key"] / "rollouts.jsonl")}
    info["scoring_interface_check"] = verify_last_logit(model, build_prompt(tokenizer, rows[0]["question"], spec))
    states, generations = [], []
    for index, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row["question"], spec)
        tokens = torch.tensor([prompt], device="cuda")
        torch.manual_seed(config["seed"] + index)
        if cached is not None:
            prior = cached[row["id"]]
            if prior["prompt_token_ids"] != prompt or prior["question"] != row["question"]:
                raise ValueError("Cached rollout prompt changed")
            generated = prior["response_token_ids"]
        else:
            with torch.inference_mode():
                generated = model.generate(
                    tokens, attention_mask=torch.ones_like(tokens), do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0,
                    max_new_tokens=config["student_rollout_max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                )[0, len(prompt):].cpu().tolist()
        generations.append({"id": row["id"], "question": row["question"],
                            "prompt_token_ids": prompt, "response_token_ids": generated,
                            "text": tokenizer.decode(generated, skip_special_tokens=False),
                            "reached_token_limit": len(generated) == config["student_rollout_max_new_tokens"]})
        for position in config["prefix_token_positions"]:
            if position >= len(generated):
                continue
            answer_prefix = visible_continuation(tokenizer, prompt, generated[:position])
            if "\ufffd" in answer_prefix:
                continue
            state = {"state_id": f"{spec['key']}:{row['id']}:{position}",
                     "question_id": row["id"], "question": row["question"],
                     "student": spec["key"], "prefix_position": position,
                     "answer_prefix_text": answer_prefix,
                     "student_input_ids": prompt + generated[:position]}
            state["student_distribution"] = score_state(
                tokenizer, model, state["student_input_ids"], info, config["native_probability_topk"],
            )
            states.append(state)
        print(json.dumps({"student": spec["key"], "completed_questions": index + 1,
                          "states": len(states)}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in [("states", states), ("rollouts", generations)]:
        with (out / f"{name}.jsonl").open("x") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    info.update({"questions": len(rows), "states": len(states), "parameter_updates": 0,
                 "gpu_admission": os.environ["STEER_OPD_GPU_ADMISSION"],
                 "created_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    dump(out / "summary.json", info)


def teacher_phase(config, out):
    tokenizer, model = load_model(config["teacher"])
    info = describe_model(tokenizer, model, config["teacher"])
    first = read_rows(ROOT / config["dev_path"])[0]
    info["scoring_interface_check"] = verify_last_logit(model, build_prompt(tokenizer, first["question"], config["teacher"]))
    summaries = {}
    with (out / "teacher_scores.jsonl").open("x") as scores, (out / "comparisons.jsonl").open("x") as compared:
        for spec in config["students"]:
            results, failures = [], []
            for state in read_rows(out / spec["key"] / "states.jsonl"):
                prompt = build_prompt(tokenizer, state["question"], config["teacher"])
                full_text = tokenizer.decode(prompt, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False) + state["answer_prefix_text"]
                teacher_ids = tokenizer.encode(full_text, add_special_tokens=False)
                roundtrip = tokenizer.decode(teacher_ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
                if roundtrip != full_text:
                    failures.append({"state_id": state["state_id"], "reason": "teacher_text_roundtrip_mismatch"})
                    continue
                target = score_state(tokenizer, model, teacher_ids, info, config["native_probability_topk"])
                scores.write(json.dumps({"state_id": state["state_id"], "distribution": target}) + "\n")
                result = {"state_id": state["state_id"], "question_id": state["question_id"],
                          "student": spec["key"], "prefix_position": state["prefix_position"],
                          **compare_distributions(target, state["student_distribution"], k=config["overlap_k"])}
                results.append(result)
                compared.write(json.dumps(result) + "\n")
                scores.flush(); compared.flush()
                print(json.dumps({"student": spec["key"], "completed_states": len(results)}), flush=True)
            numeric = [key for key, value in results[0].items()
                       if key != "prefix_position" and (value is None or
                          isinstance(value, (float, int)) and not isinstance(value, bool))] if results else []
            def averages(items):
                values = {}
                for key in numeric:
                    available = [r[key] for r in items if r[key] is not None]
                    values[key] = sum(available) / len(available) if available else None
                return values
            means = averages(results)
            by_position = {str(position): {"states": sum(r["prefix_position"] == position for r in results),
                                           "means": averages([r for r in results if r["prefix_position"] == position])}
                           for position in config["prefix_token_positions"]}
            summaries[spec["key"]] = {"spec": spec, "paired_states": len(results),
                                       "roundtrip_failures": failures, "means": means,
                                       "by_prefix_position": by_position,
                                       "native_token_ids_comparable": results[0]["native_token_ids_comparable"] if results else None}
    dump(out / "summary.json", {"teacher": info, "config": config, "students": summaries,
                                "parameter_updates": 0, "limitations": [
                                    "Literal continuation overlap depends on tokenizer segmentation.",
                                    "First-byte projection is coarse; high overlap does not establish reasoning compatibility.",
                                    "Bounds include unresolved top-k, invalid UTF-8 and special-token mass.",
                                    "Teacher re-encodes the same visible text; this is not a full text-distribution KL or OPD objective.",
                                    "Ordinary student prefixes are frozen across subsequent steering comparisons.",
                                ]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/cross_family_overlap_20261110.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student")
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Use run_on_idle_gpus.py")
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the project")
    config = json.loads(args.config.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if args.teacher:
        teacher_phase(config, out)
    else:
        spec = next(s for s in config["students"] if s["key"] == args.student)
        student_phase(config, spec, out / spec["key"])


if __name__ == "__main__":
    main()
