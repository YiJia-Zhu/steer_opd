"""Top16 prefix-compatible candidate counts and raw probability matching mass.

Only returned candidates participate. Probability mass outside Top16 is omitted;
returned logprobs retain the model normalization and are never renormalized.
"""
import torch

from steer_opd.resumable_core import digest

METRIC_VERSION = 4
TOP_K = 16
RULE_VERSION = "native_token_prefix_compatibility_v1"
P_DEFINITION = "top16_raw_probability_maximum_prefix_compatible_mass"
TOPK_TIE_POLICY = "engine_native_topk_ranks_retained_no_posthoc_boundary_expansion"


def event(codec, token):
    token = int(token)
    if token not in codec.raw:
        raise ValueError("Undefined native token in prefix compatibility")
    if token in codec.eos:
        return ("eos",)
    if token in codec.special:
        name = getattr(codec, "token_names", {}).get(token)
        return ("control", name if name is not None else codec.raw[token].decode("utf-8"))
    raw = codec.raw[token]
    return ("bytes", raw) if raw else ("empty",)


def codec_identity(codec):
    if not hasattr(codec, "_prefix_identity"):
        codec._prefix_identity = digest({"mapping": codec.mapping_sha256,
            "support": codec.support_size, "rule": RULE_VERSION,
            "events": [(i, tuple(v.hex() if isinstance(v, bytes) else v for v in event(codec, i)))
                       for i in codec.valid_ids]})
    return codec._prefix_identity


def compatible(a, b, student_codec, teacher_codec):
    if student_codec.mapping_sha256 == teacher_codec.mapping_sha256:
        return int(a) == int(b)
    return _compatible_events(event(student_codec, a), event(teacher_codec, b))


def _compatible_events(x, y):
    if x[0] == y[0] == "bytes":
        return x[1].startswith(y[1]) or y[1].startswith(x[1])
    return x[0] in ("eos", "control") and x == y


def maximum_topk_matching(student_ids, teacher_ids, student_codec, teacher_codec):
    """Augmenting paths, retaining aliases as distinct native candidates."""
    a, b = list(map(int, student_ids)), list(map(int, teacher_ids))
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise ValueError("Top-K candidates must have distinct native IDs")
    if student_codec.mapping_sha256 == teacher_codec.mapping_sha256:
        right = {token: j for j, token in enumerate(b)}
        return [(i, right[x]) for i, x in enumerate(a) if x in right]
    left_events = [event(student_codec, x) for x in a]
    right_events = [event(teacher_codec, y) for y in b]
    edges = [[j for j, y in enumerate(right_events) if _compatible_events(x, y)] for x in left_events]
    owner = [-1]*len(b)
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] < 0 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False
    for i in range(len(a)):
        augment(i, set())
    return sorted((i, j) for j, i in enumerate(owner) if i >= 0)



def _mass_overlap(left, right):
    """Exact capacity matching on a prefix forest of at most 32 candidates.

    For each subtree the maximum independent-set weight is the largest of
    its total student mass, total teacher mass and child independent weights.
    Total mass minus that weight is the bipartite minimum cut / maximum flow.
    Aliases share capacity nodes; empty ordinary tokens have no matching edges.
    """
    masses = {}
    for side, values in enumerate((left, right)):
        for label, mass in values:
            if label[0] == "empty":
                continue
            masses.setdefault(label, [0., 0.])[side] += mass
    labels = sorted(x for x in masses if x[0] == "bytes")
    parents, stack = [], []
    for index, label in enumerate(labels):
        while stack and not label[1].startswith(stack[-1][0]):
            stack.pop()
        parents.append(stack[-1][1] if stack else -1)
        stack.append((label[1], index))
    controls = [x for x in masses if x[0] != "bytes"]
    labels.extend(controls)
    parents.extend([-1]*len(controls))
    sub = [masses[x][:] for x in labels]
    independent = [0.]*len(labels)
    matched = 0.
    for i in range(len(labels)-1, -1, -1):
        a, b = sub[i]
        independent[i] = max(a, b, independent[i])
        parent = parents[i]
        if parent < 0:
            matched += a+b-independent[i]
        else:
            sub[parent][0] += a
            sub[parent][1] += b
            independent[parent] += independent[i]
    return max(0., matched)


def probability_overlap(student, teacher, student_codec, teacher_codec, *, student_positions=None, teacher_positions=None):
    """Match retained raw Top-K masses; the production metric fixes K=16."""
    from steer_opd.static_masks import validate_native_score
    for record, codec in ((student, student_codec), (teacher, teacher_codec)):
        validate_native_score(record, len(record["ids"]), codec.support_size, k=record["ids"].shape[-1])
    si = list(range(len(student["ids"]))) if student_positions is None else list(student_positions)
    ti = list(range(len(teacher["ids"]))) if teacher_positions is None else list(teacher_positions)
    if len(si) != len(ti):
        raise ValueError("Need distributions at matching visible-text prefix positions")
    a_ids, b_ids = student["ids"][si].tolist(), teacher["ids"][ti].tolist()
    a_probs = student["logps"][si].double().exp().tolist()
    b_probs = teacher["logps"][ti].double().exp().tolist()
    same = student_codec.mapping_sha256 == teacher_codec.mapping_sha256
    events = [{}, {}]
    output = []
    for ai, bi, ap, bp in zip(a_ids, b_ids, a_probs, b_probs):
        if same:
            right = dict(zip(bi, bp))
            value = sum(min(p, right.get(i, 0.)) for i, p in zip(ai, ap))
        else:
            values = []
            for ids, probs, codec, cache in zip((ai, bi), (ap, bp), (student_codec, teacher_codec), events):
                for token in ids:
                    if token not in cache:
                        cache[token] = event(codec, token)
                values.append([(cache[token], p) for token, p in zip(ids, probs)])
            value = _mass_overlap(*values)
        # Bound FP32 logprob roundoff; this does not renormalize Top16.
        output.append(min(1., value))
    return torch.tensor(output, dtype=torch.float64)
