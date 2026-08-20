"""#1064: perspective strengthening of the OA objective epigraph cut.

Two independent defects are covered here, both found while chasing the ``squfl``
family's 69-115% primal gap:

1. Both ``_extract_quadratic_terms`` and ``_extract_linear_coefficients_sparse``
   recursed once per ``+`` in a sum chain, so a long enough objective or
   constraint body raised ``RecursionError``. The caller reports that as "not
   quadratic" / "not linear" -- indistinguishable from a genuinely nonlinear
   body, so a convex separable MIQP silently lost its Hessian and its
   semicontinuity structure purely because it was long. 17 of the 1610
   MINLPLib instances hit the linear one (``sporttournament*``,
   ``edgecross*``, ``autocorr_bern*``, ``ibs2``); 2 hit the quadratic one.
2. The OA master's objective epigraph cut was the plain aggregate tangent, which
   throws away the perspective of every separable convex square over a
   semicontinuous variable (Frangioni-Gentile).
"""

import numpy as np
import pytest
from discopt import Model
from discopt._relax.perspective import (
    perspective_oa_cut_enabled,
    perspective_objective_terms,
)
from discopt._relax.problem_classifier import (
    _extract_linear_coefficients_sparse,
    _extract_quadratic_coefficients,
    dense_Q,
)
from discopt.solvers.oa import _strengthen_objective_cut_perspective


def _semicontinuous_model(n=3, u=5.0, q=None):
    """``min sum q_i x_i^2 - 4 x_i + 3 y_i`` with ``x_i <= u*y_i``."""
    q = q or [1.0] * n
    m = Model()
    xs = [m.continuous(f"x{i}", lb=0.0, ub=u) for i in range(n)]
    ys = [m.binary(f"y{i}") for i in range(n)]
    for x, y in zip(xs, ys):
        m.subject_to(x - u * y <= 0.0)
    m.minimize(sum(q[i] * xs[i] * xs[i] - 4.0 * xs[i] + 3.0 * ys[i] for i in range(n)))
    return m, xs, ys


class TestLongSumExtraction:
    """Defect 1: a long objective must not defeat quadratic extraction."""

    def test_long_separable_objective_extracts(self):
        # 3000 summed terms => a 3000-deep left-leaning ``+`` chain, well past
        # CPython's default recursion limit. Before the fix this raised
        # RecursionError, which the caller reports as "not quadratic".
        n = 3000
        m = Model()
        x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
        m.minimize(sum((i + 1) * x[i] * x[i] for i in range(n)))
        Q, c, const = _extract_quadratic_coefficients(m._objective.expression, m, n)
        Qd = dense_Q(Q) if not isinstance(Q, np.ndarray) else Q
        # ``0.5 x'Qx`` convention: a term ``k*x^2`` lands as ``Q[k,k] = 2k``.
        assert Qd.shape == (n, n)
        np.testing.assert_allclose(np.diag(Qd), 2.0 * np.arange(1, n + 1))
        assert abs(const) < 1e-12
        np.testing.assert_allclose(c, np.zeros(n), atol=1e-12)

    def test_short_chain_unchanged(self):
        """The iterative walk must agree with the hand-computed answer exactly."""
        m = Model()
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(3.0 * x * x + 2.0 * x * y - 5.0 * y + 7.0)
        Q, c, const = _extract_quadratic_coefficients(m._objective.expression, m, 2)
        Qd = dense_Q(Q) if not isinstance(Q, np.ndarray) else Q
        np.testing.assert_array_equal(Qd, np.array([[6.0, 2.0], [2.0, 0.0]]))
        np.testing.assert_array_equal(np.asarray(c), np.array([0.0, -5.0]))
        assert const == 7.0

    def test_long_linear_constraint_body_extracts(self):
        """The *linear* extractor has the same defect and the same fix.

        Reached in production via ``_relax.perspective._semicontinuity_rows``,
        which reads every row of the model to find ``x <= u*y``. A single long
        row used to abort that scan with ``RecursionError``.
        """
        n = 3000
        m = Model()
        x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
        m.subject_to(sum((i + 1) * x[i] for i in range(n)) - 12.5 <= 0.0)
        terms, const = _extract_linear_coefficients_sparse(m._constraints[0].body, m, n)
        assert len(terms) == n
        assert all(terms[i] == float(i + 1) for i in range(n))
        assert const == -12.5

    def test_linear_walk_preserves_signs_and_scales(self):
        """Nested ``-``/``*``/``/``/unary-minus must survive the work stack.

        The recursive walk carried the running ``scale`` down the call stack; the
        iterative one carries it on the stack entry. A sign dropped in that
        translation would be silent, so pin the exact coefficients.
        """
        m = Model()
        a = m.continuous("a", lb=-10.0, ub=10.0)
        b = m.continuous("b", lb=-10.0, ub=10.0)
        c = m.continuous("c", lb=-10.0, ub=10.0)
        m.subject_to(2.0 * a - (3.0 * b - c) + (-(4.0 * a)) + b / 2.0 - 6.0 <= 0.0)
        terms, const = _extract_linear_coefficients_sparse(m._constraints[0].body, m, 3)
        assert terms[0] == 2.0 - 4.0  # 2a from the head, -4a from the unary minus
        assert terms[1] == -3.0 + 0.5  # -3b from the subtraction, +b/2
        assert terms[2] == 1.0  # -(-c) from the nested subtraction
        assert const == -6.0


class TestPerspectiveTermDetection:
    def test_detects_semicontinuous_squares(self):
        m, xs, ys = _semicontinuous_model(n=3)
        terms = perspective_objective_terms(m)
        assert sorted((x, y) for x, y, _q in terms) == [(0, 3), (1, 4), (2, 5)]
        assert all(abs(q - 1.0) < 1e-12 for _x, _y, q in terms)

    def test_declines_without_an_indicator_row(self):
        m = Model()
        x = m.continuous("x", lb=0.0, ub=5.0)
        m.binary("y")  # present but never bounds x
        m.minimize(x * x)
        assert perspective_objective_terms(m) == []

    def test_declines_a_concave_term(self):
        m = Model()
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.binary("y")
        m.subject_to(x - 5.0 * y <= 0.0)
        m.minimize(-1.0 * x * x)
        assert perspective_objective_terms(m) == []

    def test_declines_a_cross_term(self):
        """A non-separable square is not a perspective candidate."""
        m = Model()
        x = m.continuous("x", lb=0.0, ub=5.0)
        z = m.continuous("z", lb=0.0, ub=5.0)
        y = m.binary("y")
        m.subject_to(x - 5.0 * y <= 0.0)
        m.minimize(x * x + x * z)
        assert perspective_objective_terms(m) == []


class TestCutStrengthening:
    """The row transform itself: valid, exact at ``y = 1``, tighter below it."""

    def _row(self, q, xbar, n_vars=2, x_col=0, y_col=1):
        # Plain tangent of ``q x^2`` at ``xbar``: ``2 q xbar x - eta <= q xbar^2``.
        coeffs = np.zeros(n_vars + 1)
        coeffs[x_col] = 2.0 * q * xbar
        coeffs[n_vars] = -1.0
        rhs = q * xbar * xbar
        return coeffs, rhs

    def test_identity_at_y_equals_one(self):
        q, xbar = 2.0, 1.5
        plain_c, plain_r = self._row(q, xbar)
        strong_c, strong_r, applied = _strengthen_objective_cut_perspective(
            self._row(q, xbar)[0], self._row(q, xbar)[1], np.array([xbar, 1.0]), 2, [(0, 1, q)]
        )
        assert applied == 1
        for x in (0.0, 0.5, 1.5, 3.0):
            pt = np.array([x, 1.0, q * x * x])
            assert abs((strong_c @ pt - strong_r) - (plain_c @ pt - plain_r)) < 1e-12

    def test_valid_at_y_equals_zero(self):
        q, xbar = 2.0, 1.5
        strong_c, strong_r, applied = _strengthen_objective_cut_perspective(
            self._row(q, xbar)[0], self._row(q, xbar)[1], np.array([xbar, 1.0]), 2, [(0, 1, q)]
        )
        assert applied == 1
        # Semicontinuity: y = 0 forces x = 0, hence eta >= 0.
        pt = np.array([0.0, 0.0, 0.0])
        assert strong_c @ pt <= strong_r + 1e-12

    def test_strictly_tighter_at_fractional_y(self):
        q, xbar = 2.0, 1.5
        plain_c, plain_r = self._row(q, xbar)
        strong_c, strong_r, _ = _strengthen_objective_cut_perspective(
            self._row(q, xbar)[0], self._row(q, xbar)[1], np.array([xbar, 1.0]), 2, [(0, 1, q)]
        )
        pt = np.array([0.75, 0.5, 0.0])
        assert (strong_c @ pt - strong_r) > (plain_c @ pt - plain_r) + 1e-9

    @pytest.mark.parametrize("xbar", [0.0, 0.25, 1.0, 4.0])
    @pytest.mark.parametrize("yval", [0.0, 1.0])
    def test_never_cuts_a_feasible_point(self, xbar, yval):
        """Every point satisfying semicontinuity and ``eta >= q x^2`` survives."""
        q = 3.0
        strong_c, strong_r, _ = _strengthen_objective_cut_perspective(
            self._row(q, xbar)[0], self._row(q, xbar)[1], np.array([xbar, 1.0]), 2, [(0, 1, q)]
        )
        checked = 0
        for x in (0.0, 0.5, 2.0, 5.0):
            if yval == 0.0 and x != 0.0:
                continue  # not feasible: y = 0 forces x = 0
            pt = np.array([x, yval, q * x * x])
            assert strong_c @ pt <= strong_r + 1e-9, (xbar, yval, x)
            checked += 1
        assert checked > 0

    def test_refuses_an_out_of_range_column(self):
        """A column past the master's own width is a layout disagreement, not a
        cut to guess at."""
        q, xbar = 2.0, 1.5
        _c, _r, applied = _strengthen_objective_cut_perspective(
            self._row(q, xbar)[0], self._row(q, xbar)[1], np.array([xbar, 1.0]), 2, [(0, 9, q)]
        )
        assert applied == 0


class TestFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("DISCOPT_PERSPECTIVE_OA_CUT", raising=False)
        assert perspective_oa_cut_enabled() is False

    def test_opt_in(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_PERSPECTIVE_OA_CUT", "1")
        assert perspective_oa_cut_enabled() is True

    def test_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_PERSPECTIVE_OA_CUT", "0")
        assert perspective_oa_cut_enabled() is False


@pytest.mark.slow
class TestEndToEnd:
    def test_flag_on_does_not_lose_the_optimum(self, monkeypatch):
        """A small semicontinuous MIQP must solve to the same optimum either way."""
        monkeypatch.setenv("DISCOPT_PERSPECTIVE_OA_CUT", "0")
        m_off, _x, _y = _semicontinuous_model(n=3, q=[1.0, 2.0, 0.5])
        r_off = m_off.solve(time_limit=30)
        monkeypatch.setenv("DISCOPT_PERSPECTIVE_OA_CUT", "1")
        m_on, _x2, _y2 = _semicontinuous_model(n=3, q=[1.0, 2.0, 0.5])
        r_on = m_on.solve(time_limit=30)
        assert r_off.objective is not None and r_on.objective is not None
        assert r_on.objective <= r_off.objective + 1e-6
        if r_off.bound is not None and r_on.bound is not None:
            # A strengthened cut may only tighten the dual bound, never loosen it
            # past the optimum it is reported against.
            assert r_on.bound <= r_on.objective + 1e-6

    def test_no_structure_is_a_no_op(self, monkeypatch):
        """Without semicontinuous squares the flag must change nothing."""
        from discopt.solvers import oa as _oa

        m = Model()
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.binary("y")
        m.subject_to(x + y >= 0.5)
        m.minimize(x * x + y)
        monkeypatch.setenv("DISCOPT_PERSPECTIVE_OA_CUT", "1")
        before = _oa._PERSPECTIVE_OA_CUT_APPLIED[0]
        r = m.solve(time_limit=30)
        assert r.objective is not None
        assert _oa._PERSPECTIVE_OA_CUT_APPLIED[0] == before
