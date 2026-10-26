"""#1958: ``canonicalize_entropy`` must honour its deadline inside one expression.

The pass consulted the deadline once per constraint, before walking it. On MINLPLib
``hadamard_9`` -- one row of 1.5 M distinct nodes (6.7 M with shared subexpressions
counted per occurrence) -- the single check passed at ~40 s and the walk then ran
~600 s past a 60 s ``time_limit``. The objective was walked with no check at all.
The walk now ticks the deadline every ``_LIFT_DEADLINE_STRIDE`` visits (as the lift
does since #1565), memoises shared nodes, and flattens each product once.
"""

from __future__ import annotations

import discopt._relax.factorable_reform as fr
import discopt.modeling as dm
import pytest


def _expires_on_second_call():
    calls = [0]

    def hook():
        calls[0] += 1
        return calls[0] >= 2

    return hook, calls


def _entropy_chain(m, n):
    x = m.continuous("x", lb=0.5, ub=2)
    zs = [m.continuous(f"z{i}", lb=0, ub=1) for i in range(n)]
    body = x * dm.log(x)  # rewritable: a finished walk changes the model
    for z in zs:
        body = body + z * z
    return body


def _has_entropy_call(expr):
    # Iterative: ``repr`` of a 3000-deep tree overflows the stack.
    from discopt.modeling.core import BinaryOp, FunctionCall, UnaryOp

    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, FunctionCall):
            if e.func_name == "entropy":
                return True
            stack.extend(e.args)
        elif isinstance(e, BinaryOp):
            stack += [e.left, e.right]
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
    return False


@pytest.mark.smoke
def test_single_row_walk_is_abandoned_mid_row():
    m = dm.Model("one_row")
    m.subject_to(_entropy_chain(m, 3000) <= 10)
    m.minimize(m._variables[0])
    hook, calls = _expires_on_second_call()
    out = fr.canonicalize_entropy(m, hook)
    # Before: one check (False) and the whole row walked and rewritten.
    assert calls[0] >= 2
    assert out is m


@pytest.mark.smoke
def test_objective_walk_is_abandoned():
    m = dm.Model("objective_only")
    m.minimize(_entropy_chain(m, 3000))
    hook, calls = _expires_on_second_call()
    out = fr.canonicalize_entropy(m, hook)
    assert calls[0] >= 2
    assert out is m


@pytest.mark.smoke
def test_unexpired_deadline_still_rewrites():
    m = dm.Model("rewrite")
    m.subject_to(_entropy_chain(m, 3000) <= 10)
    m.minimize(m._variables[0])
    out = fr.canonicalize_entropy(m, lambda: False)
    assert out is not m
    assert _has_entropy_call(out._constraints[0].body)


@pytest.mark.smoke
def test_shared_subexpressions_are_walked_once(monkeypatch):
    # e_{k+1} = e_k * 2 + e_k: 2^k occurrences of the bottom product, k + 1 distinct.
    m = dm.Model("shared")
    x = m.continuous("x", lb=0.5, ub=2)
    y = m.continuous("y", lb=0, ub=1)
    e = x * dm.exp(y)  # a product no matcher rewrites
    depth = 16
    for _ in range(depth):
        e = e * 2 + e
    m.subject_to(e <= 10)
    m.minimize(x)
    calls = [0]
    orig = fr._match_entropy_product

    def counted(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(fr, "_match_entropy_product", counted)
    out = fr.canonicalize_entropy(m)
    assert out is m
    # One call per distinct ``*`` node: the bottom product plus each level's
    # ``e_k * 2``. Before: one per occurrence (2^16 for the bottom product alone).
    assert calls[0] == 1 + depth, calls[0]
