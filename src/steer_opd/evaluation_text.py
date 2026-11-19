"""Save the decoded, unpadded benchmark responses from VERL's validation call."""
import json
import os
from pathlib import Path
from tempfile import mkstemp


FORMAT = 'question_id_text_jsonl_v1'


def response_path(run, step):
    return Path(run) / 'evaluation' / 'responses' / f'step_{step}.jsonl'


def benchmark_ids(run):
    import pyarrow.parquet as pq

    table = pq.read_table(Path(run) / 'data/benchmarks.parquet', columns=['extra_info.question_id'])
    ids = table.column(0).to_pylist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Expected unique question IDs in the frozen evaluation dataset')
    return set(ids)


def save_responses(run, step, question_ids, texts):
    expected = benchmark_ids(run)
    if len(question_ids) != len(texts) or len(texts) != len(expected) or set(question_ids) != expected:
        raise ValueError('Evaluation responses must cover each frozen benchmark question exactly once')
    target = response_path(run, step)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = mkstemp(dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for question_id, text in zip(question_ids, texts, strict=True):
                if not isinstance(text, str):
                    raise ValueError(f'Missing decoded evaluation text for {question_id}')
                stream.write(json.dumps({'question_id': question_id, 'text': text}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def responses_complete(run, step):
    path = response_path(run, step)
    if not path.is_file():
        return False
    remaining = benchmark_ids(run)
    try:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != {'question_id', 'text'} or not isinstance(row['text'], str):
                    return False
                remaining.remove(row['question_id'])
    except (ValueError, KeyError, TypeError):
        return False
    return not remaining


class EvaluationTextMixin:
    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        from steer_opd.resumable_core import ROOT

        run = ROOT / self.steer_plan['output']
        if (self.steer_plan.get('evaluation_response_format') == FORMAT
                and Path(dump_path) == response_path(run, self.global_steps).parent):
            # VERL has already removed batch padding and decoded the responses, in scoring order.
            save_responses(run, self.global_steps, reward_extra_infos_dict['eval_question_id'], outputs)
            return
        return super()._dump_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path)
