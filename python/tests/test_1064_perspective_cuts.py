"""#1064 dual half: perspective strengthening of lifted univariate squares.

The unit tests pin the two pieces that decide soundness -- which columns are
recognised as semicontinuous, and what reference point the cut is taken at --
because a wrong answer in either produces a cut that removes feasible points
without raising anything.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.perspective import perspective_reference, semicontinuous_indicators

pytestmark = pytest.mark.unit


def _vub(x_coef=1.0, y_coef=-5.0, rhs=0.0, x_lb=0.0, n=2):
    """A one-row system ``x_coef*x + y_coef*y <= rhs`` with ``x`` col 0, ``y`` col 1."""
    A = sp.csr_matrix(np.array([[x_coef, y_coef] + [0.0] * (n - 2)]))
    b = np.array([rhs])
    bounds = [(x_lb, 5.0), (0.0, 1.0)] + [(-10.0, 10.0)] * (n - 2)
    integrality = np.array([0, 1] + [0] * (n - 2), dtype=np.int32)
    return A, b, bounds, integrality


class TestSemicontinuousDetection:
    def test_the_canonical_vub_row_is_detected(self):
        assert semicontinuous_indicators(*_vub()) == {0: 1}

    def test_a_positive_rhs_is_rejected(self):
        # ``x - 5y <= 1`` leaves ``x <= 1`` at ``y = 0``; ``x`` is not switched off,
        # and a perspective cut built on it would cut off the point (1, 1, 0).
        assert semicontinuous_indicators(*_vub(rhs=1.0)) == {}

    def test_a_negative_x_lower_bound_is_rejected(self):
        # ``y = 0`` gives ``x <= 0``, not ``x = 0``: at ``x = -3`` the true square
        # is 9, and ``s >= 0`` from the cut is fine, but ``z`` is built from a
        # ratio whose sign the derivation does not cover. Refuse the pair.
        assert semicontinuous_indicators(*_vub(x_lb=-5.0)) == {}

    def test_a_positive_indicator_coefficient_is_rejected(self):
        # ``x + 5y <= 0`` says nothing about ``x`` when ``y = 0`` beyond ``x <= 0``.
        assert semicontinuous_indicators(*_vub(y_coef=5.0)) == {}

    def test_a_non_binary_partner_is_rejected(self):
        A, b, bounds, integrality = _vub()
        integrality = np.array([0, 0], dtype=np.int32)  # y is continuous now
        assert semicontinuous_indicators(A, b, bounds, integrality) == {}

    def test_an_integer_partner_with_a_wide_box_is_rejected(self):
        # ``y in [0, 7]`` integer is not an indicator: the perspective derivation
        # only covers ``y in {0, 1}``.
        A, b, bounds, integrality = _vub()
        bounds = [(0.0, 5.0), (0.0, 7.0)]
        assert semicontinuous_indicators(A, b, bounds, integrality) == {}

    def test_a_three_term_row_is_not_read_as_a_vub(self):
        A = sp.csr_matrix(np.array([[1.0, -5.0, 1.0]]))
        b = np.array([0.0])
        bounds = [(0.0, 5.0), (0.0, 1.0), (0.0, 5.0)]
        integrality = np.array([0, 1, 0], dtype=np.int32)
        assert semicontinuous_indicators(A, b, bounds, integrality) == {}

    def test_no_rows_no_indicators(self):
        assert semicontinuous_indicators(None, None, [(0.0, 1.0)], np.array([1])) == {}

    def test_an_aux_column_is_never_taken_for_a_binary(self):
        # ``integrality`` covers originals only; a lifted aux column sits past its
        # end and must not be read as a switch.
        A = sp.csr_matrix(np.array([[1.0, -5.0]]))
        b = np.array([0.0])
        bounds = [(0.0, 5.0), (0.0, 1.0)]
        assert semicontinuous_indicators(A, b, bounds, np.array([0], dtype=np.int32)) == {}


class TestPerspectiveReference:
    def test_the_reference_is_the_ratio(self):
        assert perspective_reference(0.5, 0.25) == pytest.approx(2.0)

    def test_a_vanishing_indicator_declines(self):
        assert perspective_reference(1e-9, 1e-12) is None

    def test_an_indicator_above_one_is_clamped_to_the_tangent(self):
        # An LP value slightly above 1 must not weaken the cut below the plain
        # tangent; clamping keeps the strongest valid reference.
        assert perspective_reference(3.0, 1.0 + 1e-9) == pytest.approx(3.0)

    def test_non_finite_declines(self):
        assert perspective_reference(np.nan, 0.5) is None
        assert perspective_reference(1.0, np.inf) is None


class TestCutValidity:
    """The cut ``s >= 2 z x - z^2 y`` must hold at every point of the original set."""

    @pytest.mark.parametrize("z", [0.0, 0.25, 1.0, 3.7])
    def test_no_feasible_point_is_cut(self, z):
        # The semicontinuous set: y in {0,1}; x in [0, 5] with x = 0 when y = 0;
        # s = x**2 exactly.
        checked = 0
        for y in (0.0, 1.0):
            for x in np.linspace(0.0, 5.0, 51) if y == 1.0 else [0.0]:
                s = x * x
                assert s >= 2.0 * z * x - z * z * y - 1e-9, (x, y, z)
                checked += 1
        assert checked > 0, "the validity sweep never ran"

    def test_it_dominates_the_plain_tangent(self):
        # Same reference, y < 1: the perspective right-hand side is strictly larger,
        # i.e. it is the stronger lower bound on s.
        z, x, y = 2.0, 1.0, 0.5
        assert 2 * z * x - z * z * y > 2 * z * x - z * z


class TestSeparatorIntegration:
    """Plumbing: the flag must actually change the rows the separator emits.

    This is a *synthetic* model, so it is evidence that the cut is wired up and
    tightens what the derivation says it tightens -- not evidence that it helps
    on the real class. That is what the corpus panel is for (the #727 RLT lesson:
    a mechanism validated only on a proxy can be a no-op on real instances).

    The model is the textbook semicontinuous square::

        min  s - 4x + 3y   s.t.  s >= x**2,  x <= 5y,  x in [0,5],  y in {0,1}

    whose optimum is -1 at (x, y) = (2, 1). The plain McCormick/tangent
    relaxation with ``y = x/5`` minimizes ``x**2 - 3.4x`` at ``x = 1.7``, giving
    ``-2.89``. The perspective ``s >= x**2/y`` is the convex hull here, so its
    relaxation value is the optimum ``-1`` itself.
    """

    @staticmethod
    def _model():
        from discopt import Model

        m = Model()
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.binary("y")
        m.subject_to(x - 5.0 * y <= 0.0)
        m.minimize(x * x - 4.0 * x + 3.0 * y)
        return m

    @staticmethod
    def _root_bound(model, *, perspective: bool):
        import discopt.solver_tuning as st
        from discopt._relax.mccormick_lp import MccormickLPRelaxer
        from discopt._relax.model_utils import flat_variable_bounds

        token = st.set_current(
            st.SolverTuning(perspective_cuts=perspective) if perspective else st.SolverTuning()
        )
        try:
            lb, ub = flat_variable_bounds(model)
            res = MccormickLPRelaxer(model).solve_at_node(lb, ub, time_limit=30.0)
        finally:
            st.reset_current(token)
        return res.lower_bound

    def test_the_perspective_bound_is_tighter_and_still_valid(self):
        model = self._model()
        plain = self._root_bound(model, perspective=False)
        persp = self._root_bound(self._model(), perspective=True)
        assert plain is not None and persp is not None
        # Soundness first (CLAUDE.md §1): neither may exceed the true optimum.
        assert plain <= -1.0 + 1e-6, f"plain root bound {plain} is above the optimum -1"
        assert persp <= -1.0 + 1e-6, f"perspective root bound {persp} is above the optimum -1"
        # ...and the whole point: it must be strictly tighter.
        assert persp > plain + 1e-6, (
            f"perspective did not tighten the root bound: plain={plain} persp={persp}"
        )
