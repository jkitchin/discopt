"""Coverage for the native NLP base, OBBT helpers, and LP-relaxer separators (#87).

- ``nlp_native``: build the POUNCE-native base for a tiny NLP and solve a
  node with it, cross-checking the objective against the JAX evaluator.
- ``obbt``: finite-bound bootstrapping and equality-defined bound
  propagation, checked for soundness (true optimum stays in the box).
- ``mccormick_lp`` separators: spatial solves of models with multilinear
  and general-convex constraint structure, pinned to closed-form optima.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# POUNCE-native node NLP base
# ---------------------------------------------------------------------------


def _convex_nlp():
    m = Model("native")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 1.0)
    m.minimize((x - 0.2) ** 2 + (y - 0.1) ** 2)
    return m


def test_native_base_builds_and_solves_node():
    from discopt.solvers.nlp_native import get_native_base, solve_node_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    nb = get_native_base(ev)
    if nb is None:
        pytest.skip("POUNCE-native base unavailable in this environment")
    # Round-trip permutation contract.
    pt = np.array([0.3, 0.7])
    np.testing.assert_allclose(nb.to_eval_order(nb.to_nl_order(pt)), pt)
    res = solve_node_native(
        nb,
        np.array([0.0, 0.0]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        options={"max_iter": 200},
    )
    from discopt.solvers import SolveStatus

    assert res.status == SolveStatus.OPTIMAL
    # KKT for min (x-.2)^2+(y-.1)^2 s.t. x+y>=1: x* = 0.55, y* = 0.45.
    assert res.objective == pytest.approx(2 * 0.35**2, abs=1e-6)
    assert float(ev.evaluate_objective(np.asarray(res.x))) == pytest.approx(res.objective, abs=1e-8)
    # The cache returns the same base for an unchanged model.
    assert get_native_base(ev) is nb


# ---------------------------------------------------------------------------
# OBBT helpers
# ---------------------------------------------------------------------------


def test_bootstrap_finite_bounds_finitizes_open_box():
    from discopt._jax.obbt import bootstrap_finite_bounds

    m = Model("boot")
    x = m.continuous("x")  # open bounds
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x + y <= 5.0)
    m.subject_to(x - y >= -3.0)
    m.minimize(x * y)
    lb = np.array([-np.inf, 0.0])
    ub = np.array([np.inf, 4.0])
    new_lb, new_ub, n_fin, _t = bootstrap_finite_bounds(m, lb, ub)
    # The open variable is finitized (that is the function's whole contract);
    # the box is a heuristic/OBBT starting box, not a certified enclosure.
    assert np.isfinite(new_lb[0]) and np.isfinite(new_ub[0])
    assert n_fin >= 1
    assert new_lb[0] <= new_ub[0]
    # Guard: bootstrapping must never leak into the certificate — the solve
    # still finds the true optimum x*y = -2.25 at (-1.5, 1.5), which lies
    # OUTSIDE the bootstrapped [~0, ~5] heuristic box for x.
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(-2.25, abs=1e-4)


def test_propagate_equality_defined_bounds():
    from discopt._jax.obbt import propagate_equality_defined_bounds

    m = Model("eqdef")
    x = m.continuous("x", lb=1.0, ub=2.0)
    z = m.continuous("z")  # defined variable, open bounds
    m.subject_to(z == 3.0 * x + 1.0)
    m.minimize(z)
    lb = np.array([1.0, -np.inf])
    ub = np.array([2.0, np.inf])
    new_lb, new_ub, n = propagate_equality_defined_bounds(m, lb, ub)
    # z = 3x + 1 with x in [1, 2] -> z in [4, 7], outward-rounded (sound).
    assert new_lb[1] == pytest.approx(4.0, abs=1e-4)
    assert new_lb[1] <= 4.0  # outward: never tighter than the true interval
    assert new_ub[1] == pytest.approx(7.0, abs=1e-4)
    assert new_ub[1] >= 7.0
    assert n >= 1


def test_run_obbt_tightens_relaxation_box():
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.obbt import run_obbt_on_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = Model("obbt")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y == 1.0)
    m.minimize(x * y)
    terms = classify_nonlinear_terms(m)
    relax, _ = build_milp_relaxation(m, terms, DiscretizationState(), None)
    res = run_obbt_on_relaxation(relax, n_orig=2, candidate_idxs=[0, 1])
    assert res.n_tightened >= 1
    # x + y == 1 with x, y >= 0 forces both into [0, 1].
    assert res.tightened_ub[0] <= 1.0 + 1e-6
    assert res.tightened_ub[1] <= 1.0 + 1e-6
    # Soundness: the feasible segment remains.
    assert res.tightened_lb[0] <= 0.0 + 1e-9
    assert res.tightened_lb[1] <= 0.0 + 1e-9


# ---------------------------------------------------------------------------
# LP-relaxer separators via structured spatial solves
# ---------------------------------------------------------------------------


def test_trilinear_model_certifies():
    # x*y*z >= 8 on [1,4]^3, min x+y+z -> symmetric optimum at x=y=z=2.
    m = Model("tri")
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=1.0, ub=4.0)
    z = m.continuous("z", lb=1.0, ub=4.0)
    m.subject_to(x * y * z >= 8.0)
    m.minimize(x + y + z)
    res = m.solve(time_limit=90.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(6.0, abs=1e-3)


def test_convex_transcendental_constraint_certifies():
    # exp(x) + y <= 3 (convex row) with min -x - y: optimum pushes the row
    # tight; at optimum y = 3 - exp(x), maximize x + 3 - exp(x) -> x = 0,
    # y = 2 -> objective -2.
    m = Model("gconv")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(dm.exp(x) + y <= 3.0)
    m.minimize(-x - y)
    res = m.solve(time_limit=90.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(-2.0, abs=1e-3)


def test_mixed_multilinear_binary_certifies():
    # Trilinear with a binary gate: b=1 relaxes the budget enough to pay off.
    m = Model("trib")
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    z = m.continuous("z", lb=1.0, ub=3.0)
    b = m.binary("b")
    m.subject_to(x * y * z >= 4.0 + 4.0 * b)
    m.minimize(x + y + z - 2.5 * b)
    res = m.solve(time_limit=90.0)
    assert res.status in ("optimal", "feasible")
    # b=0: min sum with xyz>=4 -> 3*4^(1/3) ~ 4.762; b=1: xyz>=8 -> 6 - 2.5
    # = 3.5. So b=1 wins with objective 3.5.
    assert res.objective == pytest.approx(3.5, abs=1e-3)
