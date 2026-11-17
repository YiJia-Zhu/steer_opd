"""CPU processes must preserve the complete metric packet and surface errors."""
import copy

import pytest
import torch

from steer_opd.compass_alignment import align_response
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.cpu_metrics import MetricExecutor, worker_count
from test_dual_overlap import codec


def assert_packet_equal(actual, expected):
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_packet_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) == type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_packet_equal(a, b)
    else:
        assert actual == expected


def fixture():
    s, t = codec(), codec(True)
    plan = align_response([1, 4, 1, 4, 0], s, {'teacher': t})
    generator = torch.Generator().manual_seed(42)
    records = []
    for key in ['q3', 'q1', 'q2']:
        def record(n):
            return {'ids': torch.stack([torch.randperm(47, generator=generator)[:16] for _ in range(n)]).int(),
                    'logps': torch.full((n, 16), -5.), 'sampled_logps': -torch.rand(n, generator=generator)}
        records.append((key, record(5), record(4)))
    adapter = AlignedMetricAdapter({key: copy.deepcopy(plan) for key, *_ in records}, s, t)
    return adapter, records


def test_spawn_preserves_all_fields_bits_order_and_reuses_pool():
    adapter, rows = fixture()
    with MetricExecutor(adapter, workers=1) as serial:
        expected = serial.compute(rows)
    with MetricExecutor(adapter, workers=2) as parallel:
        actual = parallel.compute(rows)
        first_pool = parallel.pool
        again = parallel.compute(rows[::-1])
        assert parallel.pool is first_pool
    assert_packet_equal(actual, expected)
    assert_packet_equal(again, expected[::-1])
    assert not torch.cuda.is_initialized()


def test_worker_error_propagates_without_partial_results():
    adapter, rows = fixture()
    rows[1][1]['ids'][0, 0] = 1000
    with MetricExecutor(adapter, workers=2) as parallel:
        with pytest.raises(ValueError):
            parallel.compute(rows)


def test_completed_or_single_row_work_does_not_create_a_pool(monkeypatch):
    adapter, rows = fixture()
    monkeypatch.setattr('steer_opd.cpu_metrics.ProcessPoolExecutor', lambda **kw: pytest.fail('No pool needed'))
    with MetricExecutor(adapter, workers=2) as parallel:
        assert parallel.pool is None
        assert len(parallel.compute(rows[:1])) == 1
        assert parallel.pool is None


def test_worker_count_respects_affinity_and_rejects_invalid_values(monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {0, 1})
    assert worker_count(8) == 2
    for bad in (0, -1, 'bad'):
        with pytest.raises(ValueError):
            worker_count(bad)
