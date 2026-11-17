"""Ordered, bounded CPU metric workers; GPU models remain in the parent."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from types import SimpleNamespace

import torch


_WORKER_CODECS = None


def worker_count(requested=None):
    value = os.environ.get('STEER_OPD_METRIC_WORKERS', '16') if requested is None else requested
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError('STEER_OPD_METRIC_WORKERS must be a positive integer') from error
    if count < 1:
        raise ValueError('STEER_OPD_METRIC_WORKERS must be a positive integer')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return min(count, available)


def _codec_state(codec):
    # The tokenizer/model is intentionally excluded from worker payloads.
    return {name: getattr(codec, name) for name in
            ('raw', 'eos', 'special', 'mapping_sha256', 'support_size')} | {
                'token_names': getattr(codec, 'token_names', {})}


def _initialize(student_codec, teacher_codec):
    global _WORKER_CODECS
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_CODECS = (SimpleNamespace(**student_codec), SimpleNamespace(**teacher_codec))


def _score(task):
    from steer_opd.compass_alignment import aligned_metrics, span_scores
    key, plan, student, teacher = task
    likelihoods = span_scores(plan, {'student': student, 'teacher': teacher})
    metrics = aligned_metrics(plan, student, teacher, *_WORKER_CODECS)
    return key, {**metrics, **likelihoods}


def _score_fields(record):
    fields = {name: record[name] for name in ('ids', 'logps', 'sampled_logps')}
    if any(value.device.type != 'cpu' for value in fields.values()):
        raise ValueError('CPU metric workers require CPU probability tensors')
    return fields


class MetricExecutor:
    """Reuse one spawn pool across candidates, preserving the serial row order.

    Each worker uses the existing metric functions without changing arithmetic,
    matching, diagnostics or NaNs. At most twice the worker count is submitted;
    a full candidate still commits only after every row succeeds.
    """

    def __init__(self, adapter, *, workers=None, progress=None):
        self.adapter = adapter
        self.workers = worker_count(workers)
        self.progress = progress
        self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def compute(self, records):
        """Return (key, metrics) in input order, raising on worker failure."""
        if self.workers == 1 or len(records) < 2:
            return [(key, self.adapter(key, student, teacher)) for key, student, teacher in records]
        if self.pool is None:
            count = min(self.workers, len(records))
            self.pool = ProcessPoolExecutor(max_workers=count,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
                initargs=(_codec_state(self.adapter.student_codec), _codec_state(self.adapter.teacher_codec)))
            self.workers = count
            if self.progress is not None:
                self.progress.event('cpu_metric_execution', workers=count, start_method='spawn',
                    torch_threads_per_worker=1, max_in_flight=2*count,
                    ordered_results=True, gpu_models_in_workers=False)
        records = iter(records)
        pending = deque()

        def submit():
            try:
                key, student, teacher = next(records)
            except StopIteration:
                return False
            task = (key, self.adapter.plans[key], _score_fields(student), _score_fields(teacher))
            pending.append(self.pool.submit(_score, task))
            return True

        try:
            for _ in range(2*self.workers):
                if not submit():
                    break
            result = []
            while pending:
                result.append(pending.popleft().result())
                submit()
            return result
        except BaseException:
            for future in pending:
                future.cancel()
            raise
