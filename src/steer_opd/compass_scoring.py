"""Native text-model scoring and cached Compass-style cross-tokenizer metrics."""

import os

import torch

from steer_opd.compass_alignment import METRIC, aligned_metrics, span_scores


class TextScoringModel(torch.nn.Module):
    """Expose a multimodal checkpoint's text decoder without changing weights."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.config = getattr(owner.config, "text_config", owner.config)

    @property
    def model(self):
        decoder = self.owner.model
        return getattr(decoder, "language_model", decoder)

    @property
    def lm_head(self):
        return self.owner.lm_head


def require_scoring_gpus(max_gpus=2):
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Fresh idle-GPU admission is required")
    if not 1 <= torch.cuda.device_count() <= max_gpus <= 2:
        raise RuntimeError("Text scoring must see one or two admitted project GPUs")


def load_text_scorer(model_path, codec):
    """Qwen3 and Qwen3.5 dense/MoE text paths, each with its own native support."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    require_scoring_gpus()
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    factory = AutoModelForImageTextToText if hasattr(cfg, "text_config") else AutoModelForCausalLM
    owner = factory.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="balanced" if torch.cuda.device_count() > 1 else "cuda:0", attn_implementation="sdpa")
    model = TextScoringModel(owner).eval().requires_grad_(False)
    model._opd_vocab_size = codec.support_size
    model._defined_vocab_size = codec.support_size
    if len(codec.valid_ids) != codec.support_size:
        model._opd_valid_output_ids = codec.valid_ids
    model._input_device = model.model.get_input_embeddings().weight.device
    return model


class AlignedMetricAdapter:
    metric = METRIC

    def __init__(self, plans, student_codec, teacher_codec, references):
        self.plans = plans
        self.student_codec = student_codec
        self.teacher_codec = teacher_codec
        self.references = references

    def __call__(self, key, student, teacher):
        plan = self.plans[key]
        likelihoods = span_scores(plan, {"student": student, "teacher": teacher, "reference": self.references[key]})
        return {**aligned_metrics(plan, student, teacher, self.student_codec, self.teacher_codec), **likelihoods}


class AlignedTeacherRecords(dict):
    def __init__(self, values, adapter):
        super().__init__(values)
        self.metric_adapter = adapter
