"""MO2 regression tests: AUGMECON2 fidelity.

The ``epsilon_constraint`` scalarizer is AUGMECON2 [Mavrotas & Florios 2013],
which adds two features over plain AUGMECON:

1. a **lexicographic payoff table** whose rows are Pareto-optimal, so the
   ideal/nadir range (and hence the epsilon grid) is not distorted by
   alternative optima at an anchor [Miettinen 1999]; and
2. a **bypass / jump acceleration** that uses a subproblem's slack to skip the
   epsilon-grid cells that would reproduce the same solution.

These tests are solve-backed but use tiny LP/IP models and run sub-second, so
they are ``smoke`` (run on every PR), not ``slow``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import epsilon_constraint
from discopt.mo.utils import (
    ideal_point,
    lexicographic_payoff,
    nadir_from_payoff,
    nadir_point,
)


def _alt_optimum_model():
    """Bi-objective LP with alternative optima at the f1 anchor.

    min f1 = x1, min f2 = x2 ; x1 in [0, 2], x2 in [0, 3], x1 + x2 >= 2.

    The f1 anchor (x1 = 0) leaves x2 free in [2, 3] -- alternative optima. A
    plain single-objective payoff can pick x2 = 3 (Pareto-worst), inflating the
    nadir estimate of f2 to ~3; the lexicographic payoff pins x2 = 2, giving the
    true Pareto-worst nadir(f2) = 2.
    """
    m = dm.Model("alt_opt")
    x1 = m.continuous("x1", lb=0.0, ub=2.0)
    x2 = m.continuous("x2", lb=0.0, ub=3.0)
    m.subject_to(x1 + x2 >= 2.0)
    return m, [x1, x2]


def _knapsack_step_model():
    """Bi-objective binary knapsack with a step Pareto front (flat regions)."""
    m = dm.Model("knap")
    x = m.binary("x", shape=(3,))
    p = np.array([5.0, 4.0, 3.0])
    q = np.array([2.0, 6.0, 4.0])
    f1 = p[0] * x[0] + p[1] * x[1] + p[2] * x[2]
    f2 = q[0] * x[0] + q[1] * x[1] + q[2] * x[2]
    m.subject_to(x[0] + x[1] + x[2] <= 2)
    return m, [f1, f2]


class TestLexicographicPayoff:
    @pytest.mark.smoke
    def test_lexicographic_payoff_is_pareto_optimal(self):
        # MO-2a: the lexicographic nadir matches the hand-computed Pareto-worst.
        m, objs = _alt_optimum_model()
        ideal, payoff, anchors = lexicographic_payoff(m, objs, senses=["min", "min"])
        nadir = nadir_from_payoff(payoff, senses=["min", "min"])
        # Values carry the LP tolerance and the lexicographic slack (lex_tol);
        # 1e-3 is comfortably tighter than the alternative-optimum inflation
        # (which pushes nadir(f2) to ~3) this test guards against.
        np.testing.assert_allclose(ideal, [0.0, 0.0], atol=1e-3)
        # Payoff must be the Pareto-optimal table [[0, 2], [2, 0]].
        np.testing.assert_allclose(payoff, [[0.0, 2.0], [2.0, 0.0]], atol=1e-3)
        np.testing.assert_allclose(nadir, [2.0, 2.0], atol=1e-3)

    @pytest.mark.smoke
    def test_lexicographic_differs_from_simple_under_alternative_optima(self):
        # The simple payoff (ideal_point + nadir_point) inflates nadir(f2)
        # because the f1 anchor sits on an alternative optimum; lexicographic
        # does not. This pins the *distinction* that makes it AUGMECON2.
        m_simple, objs_simple = _alt_optimum_model()
        _, anchors = ideal_point(m_simple, objs_simple, senses=["min", "min"])
        simple_nadir = nadir_point(m_simple, objs_simple, anchors, senses=["min", "min"])

        m_lex, objs_lex = _alt_optimum_model()
        _, payoff, _ = lexicographic_payoff(m_lex, objs_lex, senses=["min", "min"])
        lex_nadir = nadir_from_payoff(payoff, senses=["min", "min"])

        # Simple over-estimates the f2 nadir (alternative optimum x2 ~ 3);
        # lexicographic reports the true Pareto-worst 2.0.
        assert simple_nadir[1] > 2.0 + 1e-3
        assert lex_nadir[1] == pytest.approx(2.0, abs=1e-4)

    @pytest.mark.smoke
    def test_lexicographic_restores_objective_and_constraints(self):
        m, objs = _alt_optimum_model()
        n_cons_before = len(m._constraints)
        obj_before = m._objective
        lexicographic_payoff(m, objs, senses=["min", "min"])
        # No lexicographic tolerance constraints leak; objective restored.
        assert len(m._constraints) == n_cons_before
        assert m._objective is obj_before


class TestAugmecon2Bypass:
    @pytest.mark.smoke
    def test_bypass_skips_cells_same_front(self):
        # MO-2b: on a step front, bypass does strictly fewer solves than the
        # exhaustive grid AND returns the same nondominated set.
        def run(bypass):
            m, objs = _knapsack_step_model()
            calls = {"n": 0}
            orig = m.solve

            def counted(*a, **k):
                calls["n"] += 1
                return orig(*a, **k)

            m.solve = counted
            front = epsilon_constraint(
                m, objs, senses=["max", "max"], n_points=11, bypass=bypass, filter=False
            )
            return front, calls["n"]

        front_bypass, n_bypass = run(True)
        front_full, n_full = run(False)

        assert n_bypass < n_full, "bypass must reduce the solve count on a flat front"

        def ndset(front):
            filt = front.filtered()
            return sorted(tuple(np.round(p.objectives, 6)) for p in filt.points)

        assert ndset(front_bypass) == ndset(front_full)

    @pytest.mark.smoke
    def test_bypass_disabled_matches_full_grid(self):
        # bypass=False must be the exhaustive n_points^(k-1) grid.
        m, objs = _knapsack_step_model()
        calls = {"n": 0}
        orig = m.solve
        m.solve = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
        epsilon_constraint(m, objs, senses=["max", "max"], n_points=11, bypass=False, filter=False)
        # k=2 => n_points grid solves + k lexicographic-payoff solves (>= n_points).
        assert calls["n"] >= 11

    @pytest.mark.smoke
    def test_invalid_payoff_arg_raises(self):
        m, objs = _knapsack_step_model()
        with pytest.raises(ValueError):
            epsilon_constraint(m, objs, senses=["max", "max"], payoff="bogus")
