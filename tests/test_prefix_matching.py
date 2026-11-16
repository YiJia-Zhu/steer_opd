"""Protocol examples and independent small-graph min-cut oracles, CPU only."""
import copy
import itertools
import random

import pytest
import torch

from steer_opd.compass_alignment import align_response, aligned_metrics
from steer_opd.prefix_matching import (compatible, maximum_topk_matching,
    probability_overlap, RULE_VERSION, METRIC_VERSION)
from steer_opd.rollout_topk import tensor_hash
from test_compass_alignment import ToyCodec


def record(codec, other, probabilities, n=1):
    # Tiny pure-math fixtures may have fewer candidates; production fixes 16.
    values = torch.tensor(probabilities, dtype=torch.float64).clamp_min(1e-30)
    order = values.argsort(descending=True)
    return {'ids':torch.tensor(codec.valid_ids, dtype=torch.int32)[order].repeat(n,1),
            'logps':values[order].log().float().repeat(n,1)}


def brute_min_cut(s, t, p, q):
    """Enumerate only left vertex covers, forcing uncovered neighbors right."""
    edges = [(i, j) for i, a in enumerate(s.valid_ids) for j, b in enumerate(t.valid_ids) if compatible(a, b, s, t)]
    best = float('inf')
    for flags in itertools.product((False, True), repeat=len(p)):
        right = {j for i, j in edges if not flags[i]}
        best = min(best, sum(v for v, chosen in zip(p, flags) if chosen)+sum(q[j] for j in right))
    return best


def test_document_calculate_calcu_example_and_internal_teacher_state():
    common = [b' solve', b' simplify', b' compare', b' compute', b' check', b' find',
              b' add', b' subtract', b' multiply', b' divide', b' the', b' sum', b' result', b' answer']
    raw = {0:b'<EOS>', 1:b'calculate', 2:b'continue', **dict(enumerate(common, 3)), 17:b'late', 18:b'<CTRL>'}
    s, t = ToyCodec(raw), ToyCodec({**raw, 1:b'calcu'})
    s.special = t.special = {0, 18}
    p = [.004, .4, .2, .04, .04, .035]+[.025]*11+[.003, .003]
    q = [.004, .25, .35, .04, .04, .035]+[.025]*11+[.003, .003]
    a, b = record(s, t, p, 2), record(t, s, q, 2)
    ids = torch.tensor([1, 2]+list(range(3, 17))).repeat(2, 1)
    for score, probabilities in [(a, p), (b, q)]:
        order = torch.tensor(probabilities)[1:17].argsort(descending=True)+1
        score.update(ids=order.repeat(2,1), logps=torch.tensor(probabilities).log()[order].repeat(2,1), sampled_logps=torch.full((2,), -2.))
    plan = align_response([1, 0], s, {'teacher':t})
    assert plan['units'][0]['ranges'] == {'student':[0,1], 'teacher':[0,2]}
    metrics = aligned_metrics(plan, a, b, s, t)
    assert metrics['overlap'][0].item() == 1
    assert metrics['P_overlap'][0].item() == pytest.approx(.84, abs=2e-7)
    assert metrics['metric_version'] == METRIC_VERSION and metrics['matching_rule_version'] == RULE_VERSION
    assert torch.isnan(metrics['P_overlap'][-1])
    # Q+calcu is a different prefix, not compared to the student's Q state.
    b['logps'][1].fill_(-6.)
    changed = aligned_metrics(plan, a, b, s, t)
    torch.testing.assert_close(changed['P_overlap'], metrics['P_overlap'], equal_nan=True)


def test_augmenting_paths_and_capacity_flow_beat_exact_first_greedy():
    s = ToyCodec({1:b'a', 2:b'ab'})
    t = ToyCodec({11:b'a', 12:b'ac'})
    assert maximum_topk_matching([1,2], [11,12], s, t) == [(0,1), (1,0)]
    assert probability_overlap(record(s,t,[.5,.5]), record(t,s,[.5,.5]), s,t).item() == pytest.approx(1)


def test_only_common_initial_bytes_are_insufficient_but_real_short_candidates_can_match():
    s = ToyCodec({1:b'calculate', 2:b'continue'})
    t = ToyCodec({11:b'calendar', 12:b'late'})
    assert probability_overlap(record(s,t,[1,0]), record(t,s,[1,0]), s,t).item() == pytest.approx(0, abs=1e-12)
    short = ToyCodec({11:b'c'})
    assert len(maximum_topk_matching([1,2], [11], s,short)) == 1
    assert probability_overlap(record(s,short,[.5,.5]), record(short,s,[1]), s,short).item() == pytest.approx(1)
    # Short-token probability is a shared capacity, not reused for every edge.
    mixed = ToyCodec({11:b'c', 12:b'z'})
    assert probability_overlap(record(s,mixed,[.5,.5]), record(mixed,s,[.2,.8]), s,mixed).item() == pytest.approx(.2)


def test_eos_controls_spaces_partial_utf8_and_ordinary_empty_tokens():
    s = ToyCodec({0:b'<EOS-S>', 1:b'<A>', 2:b'', 3:b' a', 4:b'\xe4'})
    t = ToyCodec({0:b'<EOS-T>', 11:b'<B>', 12:b'a', 13:'中'.encode()})
    s.special = {0,1}; t.special = {0,11}
    assert compatible(0,0,s,t)
    assert not compatible(1,11,s,t)
    assert not compatible(2,12,s,t)
    assert not compatible(3,12,s,t)
    assert compatible(4,13,s,t)
    assert probability_overlap(record(s,t,[0,1,0,0,0]), record(t,s,[0,1,0,0]), s,t).item() == pytest.approx(0, abs=1e-12)
    assert probability_overlap(record(s,t,[1,0,0,0,0]), record(t,s,[1,0,0,0]), s,t).item() == 1
    assert probability_overlap(record(s,t,[0,0,1,0,0]), record(t,s,[0,0,1,0]), s,t).item() == pytest.approx(0, abs=1e-12)


def test_aliases_remain_distinct_for_counting_and_same_mapping_requires_native_ids():
    s = ToyCodec({1:b'a', 2:b'a', 3:b'b'})
    t = ToyCodec({11:b'a', 12:b'z'})
    assert len(maximum_topk_matching([1,2], [11], s,t)) == 1
    assert probability_overlap(record(s,t,[.6,.4,0]), record(t,s,[1,0]), s,t).item() == pytest.approx(1)
    same = ToyCodec(dict(s.raw))
    assert maximum_topk_matching([1], [2], s,same) == []
    assert probability_overlap(record(s,same,[1,0,0]), record(same,s,[0,1,0]), s,same).item() == pytest.approx(0, abs=1e-12)
    assert probability_overlap(record(s,same,[.6,.2,.2]), record(same,s,[.1,.7,.2]), s,same).item() == pytest.approx(.5)


def test_prefix_tree_solver_matches_independent_enumerated_min_cuts():
    rng = random.Random(42)
    vocabulary = [b'a',b'ab',b'ac',b'abc',b'b',b'ba',b'bb',b'bc']
    for _ in range(60):
        left = rng.sample(vocabulary, 4); right = rng.sample(vocabulary, 4)
        s = ToyCodec(dict(enumerate(left,1))); t = ToyCodec(dict(enumerate(right,11)))
        p = [rng.randint(1,7) for _ in left]; q = [rng.randint(1,7) for _ in right]
        p = [v/sum(p) for v in p]; q = [v/sum(q) for v in q]
        actual = probability_overlap(record(s,t,p), record(t,s,q), s,t).item()
        assert actual == pytest.approx(brute_min_cut(s,t,p,q), abs=2e-7)


def test_raw_top16_mass_is_not_renormalized_and_missing_tail_does_not_match():
    s = ToyCodec({i:bytes([65+i]) for i in range(20)})
    t = ToyCodec(dict(s.raw))
    # Identical Top16 retain only .48 of each model's probability mass.
    a = {'ids':torch.arange(16, dtype=torch.int32)[None], 'logps':torch.full((1,16), .03).log()}
    b = copy.deepcopy(a)
    assert probability_overlap(a,b,s,t).item() == pytest.approx(.48)
    b['ids'] = torch.arange(4,20, dtype=torch.int32)[None]
    assert probability_overlap(a,b,s,t).item() == pytest.approx(.36)
    bad = {**a,'logps':torch.zeros(1,16)}
    with pytest.raises(ValueError,match='Top-K'):
        probability_overlap(bad,b,s,t)
