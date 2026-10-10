"""#1456: the per-PASS distribution budget.

``_DISTRIBUTE_TERM_BUDGET`` (#1455) bounds one ``distribute_products`` call. A pass
that distributes every body of a model makes one call per body, so its cost was
``n_bodies x`` that budget: johnall (191 oversized bodies) spent minutes in
``has_nonconvex_integer_bilinear`` and then in ``classify_nonlinear_terms`` before
the user's ``time_limit`` was even armed. These tests pin the class, on a synthetic
model built to have the same shape (many bodies, each over the per-call budget),
and the two walkers that turned shared-DAG output into tree-sized work.
"""

from __future__ import annotations

import sys
import time

import discopt.modeling as dm
import pytest
from discopt._relax import integer_product_reform as ipr
from discopt._relax import term_classifier as tc
from discopt._relax.factorable_reform import _collect_mul_factors
from discopt._relax.milp_relaxation import (
    _expand_integer_powers_for_relaxation,
    _node_summary,
)
from discopt.modeling.core import BinaryOp, Constant


def _oversized_model(n_bodies: int):
    """``n_bodies`` constraints, each ``(x0 + ... + x7)**7``-as-a-product: 8**7 =
    2,097,152 distributed terms per body (over the 1,048,576 per-call budget),
    built as a shared DAG so construction is cheap. Integer variables, so the
    integer-bilinear detectors have real work to find under the old code."""
    m = dm.Model("pass_budget_1456")
    x = [m.integer(f"x{i}", lb=0, ub=3) for i in range(8)]
    for k in range(n_bodies):
        s = x[0]
        for v in x[1:]:
            s = s + v
        p = s
        for _ in range(6):
            p = p * s
        m.subject_to(p <= 1000 + k)
    m.minimize(x[0] + x[1])
    return m


def _forbid(monkeypatch, module, name):
    def _boom(*_a, **_k):
        raise AssertionError(f"{module.__name__}.{name} called over the per-pass budget")

    monkeypatch.setattr(module, name, _boom)


def test_model_total_charges_each_body_at_most_the_per_call_budget():
    m = _oversized_model(20)
    per_call = tc._DISTRIBUTE_TERM_BUDGET
    # 20 constraint bodies at the per-call cap plus the 2-term objective.
    assert tc.model_distribution_terms(m) == 20 * per_call + 2
    assert 20 * per_call > tc._MODEL_DISTRIBUTE_TERM_BUDGET
    assert tc.model_distribution_exceeds_budget(m, pass_name="test")


def test_integer_detectors_abstain_without_distributing(monkeypatch):
    """Fails before #1456: every detector distributed every body (20 x 1 M terms)."""
    m = _oversized_model(20)
    _forbid(monkeypatch, ipr, "distribute_products")
    checked = 0
    assert ipr.has_nonconvex_integer_bilinear(m) is False
    checked += 1
    assert ipr.has_integer_product_work(m) is False
    checked += 1
    assert ipr.has_integer_multilinear_work(m) is False
    checked += 1
    # "Nothing to do" is the answer every caller handles by leaving the model alone.
    assert ipr.reformulate_integer_bilinear(m) is m
    checked += 1
    assert ipr.reformulate_integer_multilinear(m) is m
    checked += 1
    assert checked == 5


def test_classifier_leaves_oversized_bodies_undistributed(monkeypatch):
    """Fails before #1456: the Python classifier partially distributed each body to
    the per-call budget. Over the per-pass budget an oversized body is classified
    undistributed: its product of non-constant sums is ``general_nl`` -- the sound
    answer a per-call-truncated product already got."""
    m = _oversized_model(20)
    _forbid(monkeypatch, tc, "_distribute_within_budget")
    terms = tc._classify_nonlinear_terms_python(m)
    assert len(terms.general_nl) >= 20


def test_distribute_bodies_is_distribute_products_under_the_pass_budget():
    """Bound-neutral below the per-pass budget: identical output, body by body,
    including a body over the per-call budget (partially distributed as before)."""
    m = _oversized_model(1)
    x0, x1 = m._variables[0], m._variables[1]
    bodies = [m._constraints[0].body, m._objective.expression, (x0 + 2) * (x1 - x0)]
    assert tc.model_distribution_terms(m) <= tc._MODEL_DISTRIBUTE_TERM_BUDGET
    got = list(tc.distribute_bodies(bodies, pass_name="test"))
    want = [tc.distribute_products(b) for b in bodies]
    assert len(got) == len(want) == 3
    for g, w in zip(got, want):
        assert repr(g) == repr(w)


def test_distribute_bodies_over_the_pass_budget_still_distributes_small_bodies():
    m = _oversized_model(20)
    small = m._objective.expression * (m._variables[2] + 1)
    bodies = [c.body for c in m._constraints] + [small]
    out = list(tc.distribute_bodies(bodies, pass_name="test"))
    assert repr(out[-1]) == repr(tc.distribute_products(small))
    # An oversized body comes back folded but with its products intact.
    assert repr(out[0]) == repr(tc.fold_affine_constants(bodies[0]))


def test_estimate_is_linear_in_dag_size():
    """Fails before #1456 (it tree-walked: 2**60 visits here)."""
    x = dm.Model("e").continuous("x", lb=0, ub=1)
    e = x + 1
    for _ in range(60):
        e = e + e
    t0 = time.perf_counter()
    assert tc.estimate_distributed_terms(e) == tc._TERM_CAP
    assert time.perf_counter() - t0 < 5.0


def test_collect_mul_factors_order_and_depth():
    m = dm.Model("c")
    a, b, c, d = (m.continuous(n, lb=0, ub=1) for n in "abcd")
    assert _collect_mul_factors((a * b) * (c * d)) == [a, b, c, d]
    assert _collect_mul_factors(a * (b * (c * d))) == [a, b, c, d]
    # A product chain deeper than the recursion limit (the recursive version
    # also concatenated lists: quadratic in the chain length).
    n = sys.getrecursionlimit() * 3
    p = a
    for _ in range(n):
        p = BinaryOp("*", p, Constant(1.0))
    out = _collect_mul_factors(p)
    assert len(out) == n + 1 and out[0] is a


def test_integer_power_expansion_skips_an_oversized_term():
    m = _oversized_model(1)
    body = m._constraints[0].body
    assert _expand_integer_powers_for_relaxation(body, m) is body


def test_not_affine_message_does_not_render_the_subtree():
    m = _oversized_model(1)
    assert _node_summary(m._constraints[0].body) == "BinaryOp('-')"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
