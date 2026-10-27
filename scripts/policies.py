"""Decision rule and question scores shared by every criterion.

A cost ratio r means that a missed attack window costs r and a false escalation costs 1, so the threshold is
1/(1+r). The scores of a question take a belief view (a fitted answer model bound to one start, category and
set of available questions) and return the expected information gain or the expected decision value.
"""
import hashlib
import json
import math

ACTIONS = ('file', 'network', 'process')
TOL = 1e-12
FP_COST = 1.0
COST_RATIOS = (1, 2, 5, 10, 20, 50, 100)
PRIMARY_COST_RATIO = 10
SEEDS = tuple(range(10))


def canonical(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'))


def entropy(q):
    """Binary entropy in nats; zero at the endpoints."""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -(q * math.log(q) + (1.0 - q) * math.log1p(-q))


def check_cost_ratio(r):
    if r not in COST_RATIOS or isinstance(r, bool):
        raise ValueError(f'Cost ratio {r!r} outside the sweep')
    return r


def threshold(r=PRIMARY_COST_RATIO):
    return 1 / (1 + check_cost_ratio(r))


def risk(q, r=PRIMARY_COST_RATIO):
    """Lower expected cost of the two decisions: min(r q, 1 - q)."""
    return min(float(check_cost_ratio(r)) * q, FP_COST * (1.0 - q))


def decide(q, r=PRIMARY_COST_RATIO):
    return 1 if q >= threshold(r) else 0


def random_digest(case_key, seed, action):
    """SHA256 of canonical JSON [case_key, seed, action]; integers stay integers."""
    key = [k if isinstance(k, str) else int(k) for k in case_key]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError('Seed must be an integer')
    return hashlib.sha256(canonical([key, seed, action]).encode('utf-8')).hexdigest()


def random_order(case_key, seed, actions):
    """Ascending digest; equal digests fall back to the fixed file, network, process order."""
    return sorted(actions, key=lambda a: (random_digest(case_key, seed, a), ACTIONS.index(a)))


def choose(scores):
    """Largest score; scores within TOL of the maximum resolve in fixed order."""
    best = max(scores.values())
    for a in ACTIONS:
        if a in scores and scores[a] >= best - TOL:
            return a
    raise AssertionError('No action scored')


def information_gain(view, history, q, action):
    pred = view.predict(history, action)
    if not pred:
        return 0.0
    return entropy(q) - sum(p * entropy(qa) for _, p, qa in pred)


def expected_decision_value(view, history, q, action, r=PRIMARY_COST_RATIO):
    pred = view.predict(history, action)
    if not pred:
        return 0.0
    return risk(q, r) - sum(p * risk(qa, r) for _, p, qa in pred)
