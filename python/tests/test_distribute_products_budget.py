"""Symbolic distribution must be bounded: a pre-solve scan cannot spend 44
minutes of a 20-second time limit.

``distribute_products`` expands ``(a+b)*c`` into ``a*c + b*c``, which is
exponential in how deeply sums nest inside products.  Nothing bounded it.  The
pre-solve structural scans (``has_factorable_work``, the integer-product and
quadratic detectors) distribute the raw model body purely to look for a pattern
— with no size check and no deadline: these passes run inside ``solve_model``
before branch and bound starts and neither accept nor check one, so the time
limit cannot reach them.

MINLPLib's ``johnall`` asks that path for an expansion of 3.19e9 terms.  Measured
before the budget: ``time_limit=20`` returned after 44+ minutes, having never
reached branch and bound.  After: 19.4 s, status ``time_limit``.

The budget is a backstop against a pathology, not a capability/speed trade.
Surveyed over 1610 MINLPLib instances (5,072,187 distribute calls), the largest
expansion any instance legitimately asks for is 998,002 terms (``truck``); the
cheapest pathology is ``saa_2`` at 2.18e9.  The budget sits in that gap and
truncates 2 of 1610 instances — ``johnall`` and ``saa_2``, both >2000x over.

SCOPE: this bounds ``distribute_products``, not pre-solve scanning as a whole.
Two other paths blow the same time limit by different mechanisms and are not
fixed here — a whole-model-sized ``eigvalsh`` per ``sqrt`` node in
``convexity.patterns`` (``glider400``) and ``binary_multilinear_reform._poly_add``
(``hadamard_9``, still 300 s+ against a 60 s limit with this budget in force).
Both were fixed in #1456, by bounding the work deterministically rather than by
a wall-clock deadline; see ``test_1456_psd_quadratic_support.py`` and
``test_binary_multilinear_work_budget.py``.

Truncation is ALGEBRAICALLY IDENTITY-PRESERVING (that is what
``test_budgeted_result_is_algebraically_identical`` pins), so no constraint or
objective changes meaning.  What is lost is pattern recognition, and every
consumer was audited to abstain rather than conclude — with one exception,
``_has_unbounded_nonlinear_term``, whose ``False`` *enables* a rewrite and which
therefore fails closed (``test_the_unbounded_term_guard_fails_closed``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.factorable_reform import _has_unbounded_nonlinear_term  # noqa: E402
from discopt._relax.term_classifier import (  # noqa: E402
    _DISTRIBUTE_TERM_BUDGET,
    _distribute_unbudgeted,
    distribute_products,
    distribution_exceeds_budget,
    estimate_distributed_terms,
)
from discopt.modeling.core import BinaryOp  # noqa: E402

pytestmark = pytest.mark.unit

# A COARSE BACKSTOP, not the guard.  The property these tests are really about
# is deterministic — how many terms the budget lets the walk spend — and each
# wall-clock assertion below is paired with the term-count assertion that says
# it directly.  This ceiling only has to separate "bounded" (seconds) from the
# unbudgeted behaviour (``johnall``: 44 minutes), so it is set with room for a
# loaded machine rather than tight to the fast case.
#
# It was 20.0 s and flaked: CI measured 20.07 s for
# ``test_budget_is_a_running_total_not_a_per_node_limit`` on the coverage lane,
# which runs two xdist workers with tracing on a 2-core runner — 4.1x the 4.9 s
# this machine measures for the same test unloaded.  A threshold a busy runner
# crosses is a threshold that reports load, not correctness (CLAUDE.md §9), and
# the fix for that is a deterministic assertion plus a ceiling with real margin,
# not a tighter one.  60 s is still 44x under the pathology it exists to catch.
WALL_CEILING_S = 60.0


def _blowup_model(n_factors: int = 9, width: int = 7):
    """A model whose constraint body is a product of ``n_factors`` distinct
    ``width``-term sums: ~``width ** n_factors`` distributed terms.

    This is ``johnall``'s shape, not a contrived one — a product of sums is what
    a factorable model looks like before reformulation.  Defaults give ~4.0e7
    estimated terms, ~38x the budget.
    """
    m = dm.Model("blowup")
    x = m.continuous("x", shape=(n_factors * width,), lb=0.5, ub=2.0)
    body = None
    for k in range(n_factors):
        s = x[k * width]
        for j in range(1, width):
            s = s + float(j) * x[k * width + j]
        body = s if body is None else body * s
    return m, x, body


def _evaluator(expr, model):
    """Compile *expr* once and return ``f(xs) -> float``.

    Compiled ONCE per expression, not once per sample point: a budget-truncated
    blowup body compiles to a 7.2-million-node tree that costs ~2.8 s to walk,
    so recompiling it for every point spent ~14 s re-deriving the same function.
    That is what pushed ``test_budgeted_result_is_algebraically_identical`` past
    the coverage lane's 120 s per-test timeout.
    """
    from discopt._relax.dag_compiler import compile_expression

    fn = compile_expression(expr, model)
    return lambda xs: float(np.asarray(fn(xs)))


def _additive_terms(expr) -> int:
    """How many additive terms *expr* actually carries, counted iteratively.

    The deterministic form of "the budget bounded this": a distributed result's
    term count is what the budget spends, and unlike wall-clock it does not
    depend on what else the machine is doing.  Iterative because a distributed
    blowup nests millions deep and recursion would hit the interpreter limit.
    """
    n, stack = 0, [expr]
    while stack:
        cur = stack.pop()
        if isinstance(cur, BinaryOp) and cur.op in ("+", "-"):
            stack.append(cur.left)
            stack.append(cur.right)
        else:
            n += 1
    return n


def test_the_blowup_premise_still_holds():
    """Guards every test below from going vacuous if the fixture drifts under
    the budget and they all start asserting things about ordinary expressions."""
    _m, _x, body = _blowup_model()
    est = estimate_distributed_terms(body)
    assert est > 10 * _DISTRIBUTE_TERM_BUDGET, f"fixture only estimates {est:,} terms"
    assert distribution_exceeds_budget(body)


def test_oversized_distribution_returns_promptly():
    """THE REGRESSION. Unbudgeted this does not return in any useful time."""
    _m, _x, body = _blowup_model()
    t0 = time.perf_counter()
    out = distribute_products(body)
    dt = time.perf_counter() - t0
    assert out is not None
    # Deterministic: it returned an expression the budget actually bounded, not
    # one that happened to finish inside a ceiling on a quiet machine.
    produced = _additive_terms(out)
    assert produced <= _DISTRIBUTE_TERM_BUDGET, (
        f"{produced:,} terms against a {_DISTRIBUTE_TERM_BUDGET:,}-term budget"
    )
    assert dt < WALL_CEILING_S, f"distribute_products took {dt:.1f}s"


def test_budget_is_a_running_total_not_a_per_node_limit():
    """A long sum of individually-affordable products must not add up to an
    unbounded total.  Bounding each node separately bounds nothing: 3000 terms
    each just under the limit still costs 3e9.  Caught in review of the first
    version of this fix, which had exactly that hole.
    """
    m = dm.Model("many_affordable")
    # Each summand is a product of 6 seven-term sums: 7**6 = 117,649 terms, well
    # inside the budget. Forty of them total ~4.7e6, well outside it.
    n_factors, width, n_terms = 6, 7, 40
    x = m.continuous("x", shape=(n_factors * width,), lb=0.5, ub=2.0)
    term = None
    for _ in range(n_terms):
        p = None
        for k in range(n_factors):
            s = x[k * width]
            for j in range(1, width):
                s = s + float(j + 1) * x[k * width + j]
            p = s if p is None else p * s
        term = p if term is None else term + p
    # Each summand is affordable on its own; the sum of them is not.
    per_term = estimate_distributed_terms(p)
    assert per_term <= _DISTRIBUTE_TERM_BUDGET, "summands must be individually affordable"
    assert estimate_distributed_terms(term) > _DISTRIBUTE_TERM_BUDGET, "total must exceed it"

    t0 = time.perf_counter()
    out = distribute_products(term)
    dt = time.perf_counter() - t0

    # THE ASSERTION, and it is deterministic: how many terms came out.  A
    # running total spends one shared pool over the whole expression, so the
    # result cannot carry more terms than the pool holds.  A per-node limit --
    # the hole this test exists for -- would let each of the 40 summands spend
    # 117,649 of its own and hand back 4,705,960, 4.5x over.  Measured here:
    # 941,224 terms, 0.90x the budget.
    produced = _additive_terms(out)
    assert produced <= _DISTRIBUTE_TERM_BUDGET, (
        f"{produced:,} terms distributed against a {_DISTRIBUTE_TERM_BUDGET:,}-term "
        f"budget: it is being spent per node, not as a running total "
        f"(a per-node limit yields {per_term * n_terms:,})"
    )
    # The coarse backstop, for a budget that bounded the OUTPUT without bounding
    # the WORK.  See WALL_CEILING_S: this is not the property under test.
    assert dt < WALL_CEILING_S, f"took {dt:.1f}s: the budget is not bounding the total"


# Intrinsically expensive, and not a symptom: the budget's whole job is to let
# the walk spend ~1e6 terms, so a truncated result IS a ~7.2e6-node tree and
# evaluating it costs ~2.9 s a point on this machine — 19 s for the test, which
# tracing and two xdist workers on a 2-core CI runner multiply by ~4.  The 120 s
# per-test timeout the coverage lane passes is for ordinary unit tests; this one
# declares what it actually needs rather than sampling fewer points to fit.
@pytest.mark.timeout(300)
def test_budgeted_result_is_algebraically_identical():
    """SOUNDNESS. Truncation loses recognizable structure, never meaning. If this
    fails, a constraint means something different after the budget trips and the
    whole approach is wrong."""
    m, x, body = _blowup_model(n_factors=8, width=6)
    assert distribution_exceeds_budget(body), "premise gone: fixture is under budget"
    out = distribute_products(body)

    rng = np.random.default_rng(1449)
    n = int(np.prod(x.shape))
    f_body = _evaluator(body, m)
    f_out = _evaluator(out, m)
    checks = 0
    for _ in range(5):
        xs = rng.uniform(0.5, 2.0, size=n)
        a = f_body(xs)
        b = f_out(xs)
        assert np.isfinite(a) and a != 0.0, "degenerate sample point"
        np.testing.assert_allclose(b, a, rtol=1e-9)
        checks += 1
    assert checks == 5, "evaluation loop did not run"


def test_under_budget_expressions_are_byte_for_byte_unchanged():
    """CHARACTERIZATION (bound-neutral, CLAUDE.md §5). Every expression the
    corpus actually contains is under budget, so the budget must be invisible
    there: same tree, and node identity preserved where nothing distributed."""
    m = dm.Model("ordinary")
    x = m.continuous("x", shape=(6,), lb=0.0, ub=1.0)
    bodies = [
        (x[0] + x[1]) * (x[2] + x[3]),
        (x[0] + x[1]) ** 2,
        x[0] * x[1] + x[2] / (1.0 + x[3]),
        ((x[0] + 2.0 * x[1]) * (x[2] - x[3])) * (x[4] + x[5]),
    ]
    compared = 0
    for b in bodies:
        assert not distribution_exceeds_budget(b), "fixture drifted over budget"
        assert repr(distribute_products(b)) == repr(_distribute_unbudgeted(b))
        compared += 1
    assert compared == len(bodies)
    # Identity preservation, which id()-keyed maps downstream depend on: a node
    # with nothing to distribute comes back as the same object. (Only for the
    # non-product operators -- ``_distribute_mul`` rebuilds a ``*`` node even
    # when nothing expands. That predates the budget; it is pinned here so a
    # future reader does not mistake it for something the budget introduced.)
    flat = x[0] + x[1]
    assert distribute_products(flat) is flat
    prod = x[0] * x[1]
    assert distribute_products(prod) is not prod


def test_protected_nodes_survive_the_budgeted_path():
    """``protected_squares`` holds ``id()`` values the linearizer resolves through
    its ``composite_var_map`` (#155, #358). The budgeted walk must return those
    nodes with identity intact, exactly as the unbudgeted one does — otherwise an
    over-budget body silently loses its lifts on precisely the large models this
    path exists for. (The first version of this fix descended into them.)"""
    m, x, blowup = _blowup_model()
    # The protected node must itself be over budget -- that is the case the
    # budgeted walk handles on its own rather than delegating to the unbudgeted
    # one, and so the only case that can regress.
    assert distribution_exceeds_budget(blowup), "premise gone: fixture is under budget"
    body = (x[0] + x[1]) + blowup
    protected = frozenset({id(blowup)})
    out = distribute_products(body, protected)

    seen = []

    def _walk(e):
        if id(e) in protected:
            seen.append(e)
        for attr in ("left", "right", "operand"):
            sub = getattr(e, attr, None)
            if sub is not None:
                _walk(sub)

    _walk(out)
    assert seen, "the protected node did not survive into the result at all"
    assert all(s is blowup for s in seen), "protected node was rebuilt, losing its id()"


def test_the_unbounded_term_guard_fails_closed():
    """The one caller whose ``False`` ENABLES a rewrite.

    ``_has_unbounded_nonlinear_term`` rejects denominator clearing when clearing
    would create a product with a non-finitely-bounded factor — the gear4-class
    false infeasibility.  It detects that on the DISTRIBUTED body, and
    ``_decompose_poly_product`` files any sum factor under ``extra``, which the
    degree test then skips.  So an undistributed body walks straight past the
    check.  It must answer "unbounded" when it could not look.

    Non-vacuous by construction: every variable here is finitely bounded, so a
    walk that completed would return ``False``.  A ``True`` can only come from
    the fail-closed arm.
    """
    m, _x, body = _blowup_model()
    assert distribution_exceeds_budget(body), "premise gone: fixture is under budget"
    assert _has_unbounded_nonlinear_term(body, m) is True

    # The control: the same shape, small enough to distribute, with all-finite
    # bounds -- the walk completes and correctly finds nothing unbounded. Without
    # this the assertion above would pass for a function that always says True.
    m2 = dm.Model("small")
    y = m2.continuous("y", shape=(4,), lb=0.5, ub=2.0)
    small = (y[0] + y[1]) * (y[2] + y[3])
    assert not distribution_exceeds_budget(small)
    assert _has_unbounded_nonlinear_term(small, m2) is False


@pytest.mark.slow
def test_a_blowup_model_honours_its_time_limit():
    """END TO END: the defect as the user met it. Pre-solve scanning must not
    consume the time limit. ``johnall`` overran 20 s by 44 minutes."""
    m, x, body = _blowup_model()
    m.subject_to(body <= 1e6)
    m.minimize(dm.sum([x[i] for i in range(int(np.prod(x.shape)))]))

    t0 = time.perf_counter()
    m.solve(time_limit=5.0)
    dt = time.perf_counter() - t0
    assert dt < 120.0, f"solve ran {dt:.1f}s against a 5s time_limit"
