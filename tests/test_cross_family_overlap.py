from pathlib import Path

import pytest

from steer_opd.cross_family_overlap import compare_distributions, visible_continuation


def distribution(mapping, entries):
    return {"tokenizer_mapping_sha256": mapping, "top_tokens": [
        {"token_id": index, "probability": probability,
         "continuation_hex": text.encode().hex(), "byte_event": str(text.encode()[0])}
        for index, probability, text in entries
    ]}


def test_different_ids_can_describe_identical_text_events():
    teacher = distribution("teacher_vocab", [(10, 0.6, "a"), (11, 0.4, "b")])
    student = distribution("student_vocab", [(90, 0.6, "a"), (91, 0.4, "b")])
    result = compare_distributions(teacher, student, k=2)
    assert result["paper_native_overlap_at_16"] is None
    assert result["literal_continuation_overlap_at_16"] == 1
    assert result["first_byte_overlap_lower_bound"] == 1
    assert result["first_byte_jsd_with_residual_bucket_nats"] == 0


def test_same_vocabulary_recovers_native_topk_intersection():
    teacher = distribution("same", [(10, 0.6, "a"), (11, 0.4, "b")])
    student = distribution("same", [(11, 0.6, "b"), (12, 0.4, "c")])
    assert compare_distributions(teacher, student, k=2)["paper_native_overlap_at_16"] == 0.5


def test_missing_mass_can_match_the_other_models_known_mass():
    teacher = distribution("a", [(10, 0.5, "a")])
    student = distribution("b", [(90, 1.0, "a")])
    result = compare_distributions(teacher, student, k=1)
    assert result["first_byte_overlap_lower_bound"] == 0.5
    assert result["first_byte_overlap_upper_bound"] == 1


def test_first_byte_agreement_does_not_imply_continuation_agreement():
    teacher = distribution("a", [(10, 1.0, "apple")])
    student = distribution("b", [(90, 1.0, "answer")])
    result = compare_distributions(teacher, student, k=1)
    assert result["first_byte_overlap_lower_bound"] == 1
    assert result["literal_continuation_overlap_at_16"] == 0


def test_phi_prefix_retains_contextual_leading_space():
    from transformers import AutoTokenizer

    model_path = Path(__file__).resolve().parents[1] / "assets/models/microsoft/Phi-3.5-mini-instruct"
    if not (model_path / "tokenizer.json").exists():
        pytest.skip("Local Phi tokenizer not present")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain."}], tokenize=True, return_dict=False,
        add_generation_prompt=True,
    )
    response = tokenizer.encode("We are given", add_special_tokens=False)
    assert tokenizer.decode(response) == "We are given"
    assert visible_continuation(tokenizer, prompt, response) == " We are given"
