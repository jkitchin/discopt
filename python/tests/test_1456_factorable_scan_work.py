"""#1456 follow-up: the factorable-reform structural scan does linear work.

``has_factorable_work`` distributes each body and walks the result. The walk
used to be a *tree* walk over a DAG -- a budget-truncated distributed body
shares every product prefix -- and decomposed every ``*`` node and then every
sub-product again (quadratic in a product's length). On the #1456 blowup fixture
(``test_distribute_products_budget._blowup_model``) that was 9.9 M
decompositions and ~40 s of a 45 s solve against ``time_limit=5``, all spent
before the time limit was first consulted.

These tests pin (1) the work: each DAG node's variable slot is resolved once,
not once per product that contains it; (2) that the faster scan answers
exactly what the old one did; and (3) that the scan's whole-model distribution
spends from the deterministic per-pass budget and abstains, logged, when it runs
out.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import logging  # noqa: E402
import random  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax import factorable_reform as fr  # noqa: E402
from discopt._relax import term_classifier as tc  # noqa: E402
from discopt.modeling.core import BinaryOp, FunctionCall, UnaryOp  # noqa: E402

pytestmark = pytest.mark.unit


def _product_of_sums(n_factors: int, width: int):
    m = dm.Model("pos")
    x = m.continuous("x", shape=(n_factors * width,), lb=0.5, ub=2.0)
    body = None
    for k in range(n_factors):
        s = x[k * width]
        for j in range(1, width):
            s = s + float(j) * x[k * width + j]
        body = s if body is None else body * s
    return m, x, body


def _dag_nodes(expr) -> int:
    seen: set[int] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, BinaryOp):
            stack += [e.left, e.right]
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, FunctionCall):
            stack += list(e.args)
    return len(seen)


def test_the_mixed_product_scan_resolves_each_node_once(monkeypatch):
    """THE REGRESSION, as a deterministic work count rather than a timing.

    Unfixed, the scan resolved a variable slot once per (``*`` node, leaf below
    it) pair of the *tree*: 27 M resolutions on the blowup fixture. Fixed, it
    resolves each distinct DAG node at most once.
    """
    m, _x, body = _product_of_sums(5, 5)  # 3125 terms: fully distributed
    dist = tc.distribute_products(body)
    n_nodes = _dag_nodes(dist)

    calls = 0
    real = fr._leaf_index_and_exp

    def counting(e, model):
        nonlocal calls
        calls += 1
        return real(e, model)

    monkeypatch.setattr(fr, "_leaf_index_and_exp", counting)
    assert fr._scan_for_mixed_product(dist, m) is False  # 5 distinct sums: no repeats
    assert calls > 0, "probe never fired"
    assert calls <= n_nodes, f"{calls} slot resolutions over a {n_nodes}-node DAG"


# --- (2) exactly the old verdict -------------------------------------------------


def _reference_scan_for_mixed_product(expr, model) -> bool:
    """The pre-fix implementation, verbatim: decompose every ``*`` node."""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            decomp = fr._decompose_poly_product(expr, model)
            if decomp is not None:
                _coeff, powers, extra = decomp
                if not extra and fr._needs_lift(powers):
                    return True
        return _reference_scan_for_mixed_product(
            expr.left, model
        ) or _reference_scan_for_mixed_product(expr.right, model)
    if isinstance(expr, UnaryOp):
        return _reference_scan_for_mixed_product(expr.operand, model)
    return False


def _random_expr(rng: random.Random, leaves, depth: int, shared: list):
    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if shared and r < 0.2:
            return rng.choice(shared)  # DAG sharing, as distribution produces
        if r < 0.35:
            return float(rng.choice([2.0, -1.5, 3.0]))
        return rng.choice(leaves)
    op = rng.choice(["*", "*", "*", "+", "-", "pow2", "pow3", "powh", "exp", "neg"])
    a = _random_expr(rng, leaves, depth - 1, shared)
    if op in ("*", "+", "-"):
        b = _random_expr(rng, leaves, depth - 1, shared)
        if isinstance(a, float) and isinstance(b, float):
            return a
        out = a * b if op == "*" else (a + b if op == "+" else a - b)
    elif isinstance(a, float):
        return a
    elif op == "pow2":
        out = a**2
    elif op == "pow3":
        out = a**3
    elif op == "powh":
        out = a**0.5
    elif op == "exp":
        out = dm.exp(a)
    else:
        out = -a
    shared.append(out)
    return out


def test_the_scan_answers_exactly_what_the_old_scan_answered():
    """Bound-neutral by construction: same verdict on every expression, raw and
    distributed, across random products with repeats, powers, constants,
    transcendental factors and shared subexpressions."""
    m = dm.Model("diff")
    v = m.continuous("v", shape=(4,), lb=0.5, ub=2.0)
    w = m.continuous("w", lb=0.5, ub=2.0)
    leaves = [v[0], v[1], v[2], v[3], w]
    rng = random.Random(1456)
    compared = {True: 0, False: 0}
    for _ in range(600):
        e = _random_expr(rng, leaves, depth=5, shared=[])
        if isinstance(e, float):
            continue
        for form in (e, tc.distribute_products(e)):
            want = _reference_scan_for_mixed_product(form, m)
            assert fr._scan_for_mixed_product(form, m) is want, str(form)
            compared[want] += 1
    # Non-vacuous: both answers occur many times.
    assert compared[True] >= 50 and compared[False] >= 50, compared


def test_a_pure_subproduct_under_an_impure_product_is_found():
    """The case a 'decompose only maximal products' shortcut would miss: the
    root has a transcendental factor, the sub-product ``x*(x*y)`` needs a lift."""
    m = dm.Model("sub")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    z = m.continuous("z", lb=0.5, ub=2.0)
    e = (x * (x * y)) * dm.exp(z)
    assert _reference_scan_for_mixed_product(e, m) is True
    assert fr._scan_for_mixed_product(e, m) is True


# --- (3) the per-pass budget ----------------------------------------------------


def _late_lift_model():
    """Three bodies; only the last one has factorable work, and only after
    distribution (``x*(x*y + z)`` -> ``x*x*y + x*z``)."""
    m = dm.Model("late")
    a = m.continuous("a", shape=(4,), lb=0.5, ub=2.0)
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    z = m.continuous("z", lb=0.5, ub=2.0)
    m.subject_to((a[0] + a[1]) * (a[2] + a[3]) <= 10.0)
    m.subject_to((a[0] + a[2]) * (a[1] + a[3]) <= 10.0)
    m.subject_to(x * (x * y + z) <= 10.0)
    m.minimize(a[0])
    return m


def test_distribute_charged_is_distribute_products():
    _m, _x, body = _product_of_sums(3, 3)
    budget = tc.pass_distribution_budget()
    out = tc.distribute_charged(body, budget)
    assert str(out) == str(tc.distribute_products(body))
    assert budget.spent(tc.DISTRIBUTED_TERMS) == tc.estimate_distributed_terms(body) == 27


def test_the_scan_abstains_when_the_pass_budget_runs_out(monkeypatch, caplog):
    m = _late_lift_model()
    # Control: with the shipped budget the scan reaches the third body.
    assert fr.has_factorable_work(m) is True

    # The objective costs 1 term and the first two bodies 5 each (4 products and
    # the constant); the third (3 terms) does not fit in 13.
    monkeypatch.setattr(tc, "_MODEL_DISTRIBUTE_TERM_BUDGET", 13)
    with caplog.at_level(logging.WARNING, logger="discopt._relax.factorable_reform"):
        assert fr.has_factorable_work(m) is False
    msgs = [r.getMessage() for r in caplog.records if "per-pass budget" in r.getMessage()]
    assert msgs and "13-term" in msgs[0] and "after 11 terms" in msgs[0], msgs

    # And the budget is exactly what decided it: one more term and it fits.
    monkeypatch.setattr(tc, "_MODEL_DISTRIBUTE_TERM_BUDGET", 14)
    assert fr.has_factorable_work(m) is True


def test_the_shipped_pass_budget_is_the_whole_model_one():
    budget = tc.pass_distribution_budget()
    assert budget.remaining(tc.DISTRIBUTED_TERMS) == tc._MODEL_DISTRIBUTE_TERM_BUDGET
