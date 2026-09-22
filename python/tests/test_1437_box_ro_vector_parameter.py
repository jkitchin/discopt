"""Regression suite for #1437: the box robust counterpart on a VECTOR uncertain
parameter.

Two mathematically identical spellings of the same uncertain row behaved
completely differently. Measured before this fix, on

    max 3*x0 + 2*x1   s.t.  (a + xi)'x <= 8 for all |xi_j| <= delta_j,  x in [0,5]^2
    a = (1, 2),  delta = (0.3, 0.5)

whose true robust optimum -- from enumerating the 4 vertices of the box and
solving with scipy -- is **16.2**:

    dm.sum(a * x) <= 8        -> optimal, 16.2, worst-case residual 0.0   (correct)
    a @ x <= 8                -> optimal, 16.2, worst-case residual 0.0   (correct)
    a[0]*x[0] + a[1]*x[1] <= 8 -> CRASH on every route

The element-indexed spelling took the bilinear branch, where two defects
compounded:

1. A parameter that does not OCCUR in an expression still entered the branch.
   The coefficient is a finite difference ``g(p+1) - g(p)``; with ``p`` absent
   both sides are the same expression, so the difference is identically zero but
   still *syntactically* variable-bearing, and ``_contains_variable`` said True.
   That put a spurious aux variable, two degenerate rows (``(e - e) - t <= 0``,
   i.e. ``t >= 0`` and nothing more) and a penalty into the OBJECTIVE of a model
   whose uncertain parameter appeared only in a constraint.

2. On the bilinear branch a vector parameter got ONE scalar aux, then multiplied
   by the whole ``delta`` vector: ``Constant(delta) * t``. That is a shape-(k,)
   term inside a scalar row/objective, which the LP route rejects with
   ``jax.grad ... Output had shape: (2,)`` and POUNCE with ``cannot reshape array
   of shape (2,) into shape (1,)``. It was also wrong in principle: the box
   counterpart of ``sum_j coeff_j(x) xi_j`` is ``sum_j delta_j |coeff_j(x)|``,
   which needs one aux PER COMPONENT, not one aggregate standing for
   ``sum_j coeff_j``.

Scalar parameters (k = 1) are unaffected by construction, which is why
``test_robust_solve.py`` -- scalar throughout -- never caught it, and why
``test_robust_counterpart.py::test_vector_parameter`` did not either: it writes
the row as ``dm.sum(c * x)``, the spelling that takes the correct sign-tracking
branch, and asserts on constants without ever solving.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart
from scipy.optimize import linprog

A_NOM = np.array([1.0, 2.0])
DELTA = np.array([0.3, 0.5])
COST = np.array([3.0, 2.0])
RHS = 8.0
UB = 5.0

#: Every vertex of the box; the worst case of a linear row over a box is attained
#: at one of them, so feasibility here is feasibility over the whole set.
VERTICES = np.array([A_NOM + DELTA * np.array(z) for z in itertools.product([-1.0, 1.0], repeat=2)])


def _true_robust_optimum(delta):
    verts = np.array(
        [A_NOM + delta * np.array(z) for z in itertools.product([-1.0, 1.0], repeat=2)]
    )
    ref = linprog(
        -COST, A_ub=verts, b_ub=[RHS] * len(verts), bounds=[(0.0, UB)] * 2, method="highs"
    )
    assert ref.status == 0, ref
    return -ref.fun, verts


def _build(row, delta):
    m = dm.Model("ro1437")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=UB)
    a = m.parameter("a", A_NOM)
    m.maximize(float(COST[0]) * x[0] + float(COST[1]) * x[1])
    row(m, a, x)
    RobustCounterpart(m, BoxUncertaintySet(a, delta)).formulate()
    return m


_ELEMENT_INDEXED = pytest.param(
    lambda m, a, x: m.subject_to(a[0] * x[0] + a[1] * x[1] <= RHS), id="element-indexed"
)
_SUM = pytest.param(lambda m, a, x: m.subject_to(dm.sum(a * x) <= RHS), id="dm.sum(a*x)")
_MATMUL = pytest.param(lambda m, a, x: m.subject_to(a @ x <= RHS), id="a@x")
ALL_SPELLINGS = [_ELEMENT_INDEXED, _SUM, _MATMUL]


class TestEverySpellingGivesTheRobustOptimum:
    """The three spellings are the same mathematics and must give the same answer.
    Before #1437 the first one crashed instead."""

    @pytest.mark.parametrize("row", ALL_SPELLINGS)
    @pytest.mark.parametrize("delta", [DELTA, 0.4], ids=["per-component-delta", "scalar-delta"])
    def test_optimum_and_robustness(self, row, delta):
        want, verts = _true_robust_optimum(
            delta if isinstance(delta, np.ndarray) else np.full(2, float(delta))
        )
        m = _build(row, delta)
        r = m.solve(time_limit=40)
        assert r.objective is not None, f"no solution ({r.status})"

        xv = np.asarray(r.x["x"], dtype=float).ravel()[:2]
        # R1 -- feasible at EVERY realization, not just the nominal one. This is
        # what "robust" means, and it is the half a too-permissive counterpart
        # gets wrong.
        worst = float(np.max(verts @ xv - RHS))
        assert worst <= 1e-6, (
            f"x={xv} violates the row by {worst:.3e} at a realizable xi -- not robust "
            "over the declared set"
        )
        # R2 -- and not MORE conservative than the set requires, which is the
        # other direction of wrong (it silently costs the user objective).
        assert r.objective == pytest.approx(want, abs=1e-4), (
            f"objective {r.objective}, the vertex-enumeration robust optimum is {want}"
        )

    def test_the_three_spellings_agree(self):
        """Pairwise, not just against the oracle: if they ever diverge the API has
        two different meanings for one model."""
        vals = []
        for row in (_ELEMENT_INDEXED.values[0], _SUM.values[0], _MATMUL.values[0]):
            r = _build(row, DELTA).solve(time_limit=40)
            assert r.objective is not None
            vals.append(r.objective)
        assert vals[0] == pytest.approx(vals[1], abs=1e-6)
        assert vals[1] == pytest.approx(vals[2], abs=1e-6)


class TestTheCounterpartIsWellFormed:
    def test_the_objective_keeps_its_shape(self):
        """The shape-(k,) penalty is what crashed the LP route several layers away,
        with nothing in the message pointing back at the reformulation. Assert the
        structural property directly so a regression is caught at formulate time."""
        m = _build(_ELEMENT_INDEXED.values[0], DELTA)
        # An objective that still carried a shape-(2,) term would not survive a
        # solve; assert the value it produces is a scalar.
        r = m.solve(time_limit=40)
        assert r.objective is not None
        assert np.ndim(r.objective) == 0, f"objective is not scalar: {r.objective!r}"

    def test_no_aux_is_created_for_a_parameter_absent_from_the_objective(self):
        """``a`` appears only in the CONSTRAINT here, so the objective needs no
        robustification at all. Before #1437 it acquired an aux variable and two
        degenerate rows anyway."""
        m = dm.Model("obj_untouched")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=UB)
        a = m.parameter("a", A_NOM)
        m.maximize(float(COST[0]) * x[0] + float(COST[1]) * x[1])
        m.subject_to(a[0] * x[0] + a[1] * x[1] <= RHS)
        before = str(m._objective.expression)
        RobustCounterpart(m, BoxUncertaintySet(a, DELTA)).formulate()
        after = str(m._objective.expression)
        assert after == before, (
            f"the objective was rewritten though the uncertain parameter never "
            f"appears in it:\n  before: {before}\n  after:  {after}"
        )

    def test_one_aux_per_uncertain_component(self):
        """``sum_j delta_j |coeff_j|`` needs k auxiliaries, not one. With a single
        aggregate aux the counterpart protects against the wrong set."""
        m = _build(_ELEMENT_INDEXED.values[0], DELTA)
        n_abs = sum(1 for v in m._variables if "abs" in v.name)
        assert n_abs == 2, f"expected one absolute-value aux per component (2), got {n_abs}"

    def test_a_zero_delta_component_gets_no_aux(self):
        """A component with no uncertainty contributes no penalty; emitting one
        would add rows that can only ever be slack."""
        m = _build(_ELEMENT_INDEXED.values[0], np.array([0.3, 0.0]))
        n_abs = sum(1 for v in m._variables if "abs" in v.name)
        assert n_abs == 1, f"expected 1 aux for one uncertain component, got {n_abs}"


class TestScalarParametersAreUnchanged:
    """k = 1 is the case the existing suite covers; it must stay bit-identical."""

    def test_ben_tal_scalar_case_still_holds(self):
        """min c*x s.t. x >= d, c = 10 +/- 2, d = 5 +/- 1. Worst case 12 * 6 = 72
        (the fixture from ``test_robust_solve.py::test_scalar_cost_and_demand``)."""
        m = dm.Model("scalar")
        x = m.continuous("x", lb=0, ub=100)
        c = m.parameter("c", value=10.0)
        d = m.parameter("d", value=5.0)
        m.minimize(c * x)
        m.subject_to(x >= d)
        RobustCounterpart(
            m, [BoxUncertaintySet(c, delta=2.0), BoxUncertaintySet(d, delta=1.0)]
        ).formulate()
        r = m.solve(time_limit=40)
        assert r.x["x"] == pytest.approx(6.0, abs=0.1)
        assert r.objective == pytest.approx(72.0, abs=0.5)
