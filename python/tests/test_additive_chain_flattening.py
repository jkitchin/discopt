"""Builtin ``sum()`` chains must not blow the AD tape's depth limit.

``dm.sum`` emits a flat ``SumOverExpression``, but Python's builtin
``sum(terms)`` -- what a user writing a least-squares objective or a mass
balance reaches for -- folds left-associatively into nested ``BinaryOp`` nodes
whose depth is the term count. Each lowered to a binary tape node, so tape depth
grew with the model: measured at N+2 against ``dm.sum``'s constant 3.

Past ``NlExpr.max_depth`` (10 000) pounce refuses with a ``ValueError``, and that
refusal was not a clean degrade:

* ``try_build`` catches only ``UnsupportedForTape``, so the ``ValueError``
  propagated and ``Model.solve()`` raised outright -- the documented JAX
  fallback never engaged;
* on the paths that survived it, the incumbent-verification snapshot failed and
  **the false-primal guard was disabled for that solve**. A modelling idiom
  silently switching off a correctness guard is the worst available outcome
  (CLAUDE.md §1).

``_nl_expr_compiler`` now flattens a maximal ``+``/``-`` chain into one n-ary
``E.sum``, which is depth 2 whatever the term count. The flattening is
bound-neutral: ``E.sum`` accumulates in list order and IEEE ``a - b`` rounds
identically to ``a + (-b)``, so the tape is bit-identical to the chain.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _chain_model(n, how, nonlinear=False):
    m = dm.Model(f"chain_{n}_{how}")
    x = m.continuous("x", shape=(5,), lb=0.3, ub=3.0)
    if nonlinear:
        terms = [dm.exp(x[i % 5] * 0.1) * float(i + 1) - x[(i + 1) % 5] for i in range(n)]
    else:
        terms = [x[i % 5] * float(i + 1) for i in range(n)]
    m.minimize(dm.sum(terms) if how == "dmsum" else sum(terms))
    return m


@pytest.mark.parametrize("n", [1000, 12000, 50000])
def test_builtin_sum_chain_stays_shallow(n):
    """Tape depth must not grow with the term count.

    12 000 and 50 000 both raised ``ValueError: expression nesting would reach
    depth 10001`` before the flattening.
    """
    from discopt._nl_expr_compiler import compile_to_nl_expr

    m = _chain_model(n, "builtin")
    node = compile_to_nl_expr(m._objective.expression, m)
    assert node.depth <= 4, f"tape depth {node.depth} for {n} terms"


def test_deep_chain_model_solves_instead_of_raising():
    """A model with a >max_depth chain must solve, not raise.

    Regression guard for the propagating ``ValueError``: this used to take down
    ``Model.solve()`` and disable the false-primal guard.
    """
    m = dm.Model("deep")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=5.0)
    m.subject_to(sum(x[i % 3] * float(i + 1) for i in range(12000)) <= 1e9, name="c")
    m.minimize(x[0])
    res = m.solve(time_limit=60)
    assert res.status in ("optimal", "feasible"), f"status={res.status}"


def test_tape_builds_for_deep_chain():
    """``try_build`` must return an evaluator, not propagate."""
    from discopt._tape_nlp_evaluator import try_build

    m = _chain_model(12000, "builtin")
    ev = try_build(m)
    assert ev is not None, "tape refused a model it should now represent"


@pytest.mark.parametrize("n", [2, 3, 5, 50, 500])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_flattening_is_bound_neutral(n, nonlinear):
    """Flattened and ``dm.sum`` forms of the same maths must agree exactly.

    Both routes now build one n-ary ``E.sum`` over the same terms in the same
    order, so agreement is exact rather than within a tolerance. ``n`` spans
    below and above the flattening threshold.
    """
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    ev_a = TapeNLPEvaluator(_chain_model(n, "dmsum", nonlinear))
    ev_b = TapeNLPEvaluator(_chain_model(n, "builtin", nonlinear))
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(3):
        x = rng.uniform(0.35, 2.9, size=ev_a.n_variables)
        assert ev_a.evaluate_objective(x) == ev_b.evaluate_objective(x)
        np.testing.assert_array_equal(
            np.asarray(ev_a.evaluate_gradient(x)),
            np.asarray(ev_b.evaluate_gradient(x)),
        )
        compared += 1 + int(np.asarray(ev_a.evaluate_gradient(x)).size)
    assert compared > 0, "probe compared nothing"
