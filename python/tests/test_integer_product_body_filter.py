"""The integer-product detectors skip bodies with no integer-factor variable.

``truck`` (MINLPLib): 5,007 continuous variables, five implied integers, and an
objective that distributes to ~1M terms (just under the per-call budget). None of
the implied integers appears in it, yet ``has_nonconvex_integer_bilinear``
distributed it (7 s) and walked the result (34 s) before the solve's 60 s limit
was armed, so the run ended at 72 s with nothing found. Every hit of every
detector, and every rewrite, needs an integer factor (``_int_factor_range``), so a
body that references no such variable is skipped *before* distribution -- an
exact filter, never a change of answer.
"""

from __future__ import annotations

import discopt.modeling as dm
from discopt._relax import integer_product_reform as ipr


def _product_of_sums(m, xs):
    s = xs[0]
    for v in xs[1:]:
        s = s + v
    return s * s


def _counting_distribute(monkeypatch):
    calls: list[int] = []
    real = ipr.distribute_products

    def spy(expr, *a, **k):
        calls.append(id(expr))
        return real(expr, *a, **k)

    monkeypatch.setattr(ipr, "distribute_products", spy)
    return calls


def test_body_without_integer_factor_is_not_distributed(monkeypatch):
    """Fails before the filter: the continuous-only objective was distributed."""
    m = dm.Model("filter_skip")
    xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(30)]
    y = m.integer("y", lb=0, ub=7)
    z = m.integer("z", lb=0, ub=7)
    m.subject_to(y * z <= 20)  # the one body with integer factors
    m.minimize(_product_of_sums(m, xs))
    calls = _counting_distribute(monkeypatch)
    _ = list(ipr._bodies(m))
    assert len(calls) == 1, calls
    assert calls[0] == id(m._constraints[0].body)
    # ...and the detector still finds the integer bilinear term.
    assert ipr.has_nonconvex_integer_bilinear(m) is True


def test_no_integer_variables_distributes_nothing(monkeypatch):
    m = dm.Model("filter_none")
    xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(10)]
    m.subject_to(xs[0] * xs[1] <= 0.5)
    m.minimize(_product_of_sums(m, xs))
    calls = _counting_distribute(monkeypatch)
    assert ipr.has_nonconvex_integer_bilinear(m) is False
    assert ipr.has_integer_product_work(m, frozenset(), multilinear=True) is False
    assert ipr.has_integer_multilinear_work(m) is False
    assert ipr._participation(m, frozenset()) == {}
    assert calls == []


def test_reference_through_any_node_type_is_seen():
    """The filter walks every Expression-valued attribute, so a reference buried
    under a function call, a unary op or a sum-over is still found."""
    m = dm.Model("filter_nested")
    x = m.continuous("x", lb=0, ub=1)
    y = m.integer("y", lb=0, ub=7)
    cands = ipr._integer_factor_vars(m, frozenset())
    assert cands == frozenset({y._index})
    checked = 0
    for expr in (dm.exp(-(x * y)), dm.sum([x, x * y, x]), dm.log(1 + dm.exp(x) * y)):
        assert ipr._references_any(expr, cands)
        checked += 1
    assert not ipr._references_any(dm.exp(-(x * x)), cands)
    checked += 1
    assert checked == 4


def test_implied_integer_counts_as_candidate():
    m = dm.Model("filter_implied")
    x = m.continuous("x", lb=0, ub=5)
    w = m.continuous("w", lb=0, ub=5)
    implied = frozenset({(x._index, 0)})
    assert ipr._integer_factor_vars(m, implied) == frozenset({x._index})
    m.subject_to(x * w <= 3)
    m.minimize(w)
    # x is implied-integer: the bilinear product is visible with, not without, it.
    assert ipr._participation(m, implied) == {(x._index, 0): 1}
    assert ipr._participation(m, frozenset()) == {}
