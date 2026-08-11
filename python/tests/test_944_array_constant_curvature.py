"""#944: a convex QP written with the vectorized DAE API never terminated.

``test_dae.py::TestOptimalControl::test_minimum_energy`` — a 100-variable convex
QP with 79 linear rows and no integer variables — ran past an 800 s pytest
timeout on ``main``, and the same construct in ``docs/notebooks/tutorial_dae.ipynb``
hit an ``nbclient`` ``CellTimeoutError`` after 600 s. Both were routed to spatial
McCormick Branch-and-Bound (~9k nodes, never certifying) instead of to the single
convex NLP solve the model deserves.

Root cause: the curvature walker's constant-scaling rules keyed on
``_is_scalar_const``, which requires ``ndim == 0``. Every constant produced by
the *vectorized* modeling API is an array, so the collocation row

    (x @ A) - h ⊙ (-x[:, 1:] + u[:, None])          # h.shape == (nfe, 1)

fell to ``Curvature.UNKNOWN`` at the ``h ⊙ (...)`` node — an affine body the
walker could not see was affine. ``classify_model`` then reported the model
nonconvex, the convex-NLP fast path in ``solver.solve_model`` declined it, and
the sound-but-hopeless spatial path took over.

This is a *class* defect, not a DAE one: every array coefficient in the model —
element widths, collocation weights, quadrature weights, any ``numpy`` array
multiplying an expression — was invisible to the rule.

Soundness argument for the fix. For an elementwise (broadcast) product
``c ⊙ f`` with ``c`` constant, entry ``k`` is ``c_i * f_j``, a *scalar* constant
times a scalar expression:

* ``f`` affine  ⇒ every entry affine, for ANY ``c`` (mixed signs included);
* ``f`` convex and every ``c`` entry ≥ 0  ⇒ every entry convex;
* ``f`` convex and every ``c`` entry ≤ 0  ⇒ every entry concave;
* ``f`` non-affine and ``c`` mixed in sign ⇒ refuse (UNKNOWN).

Zero entries never spoil a verdict (``0 * f`` is affine, hence both convex and
concave). Division carries the same rule with the extra requirement that no
entry sits at or within 1e-30 of zero.

The second defect fixed here is in the Rust bridge: ``convert_index_spec``
extracted every scalar subscript as ``usize``, so ``x[-1]`` raised
``OverflowError: can't convert negative int to unsigned`` and aborted
``model_to_repr`` outright. ``classify_problem`` catches that and falls back to
``NLP``/``MINLP``, so every model using a negative index silently lost its
LP/QP/MILP classification. This is the same family as #941 (which fixed the
Python-side flat-slot resolvers) at the one site #941 did not reach.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.convexity import classify_model  # noqa: E402
from discopt._relax.convexity.lattice import Curvature  # noqa: E402
from discopt._relax.convexity.rules import classify_expr  # noqa: E402
from discopt._relax.nlp_evaluator import cached_evaluator  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402

# Load gate (CLAUDE.md §8): assert both the file under test and a marker unique
# to this version, so a stale install cannot pass this file vacuously.
assert "discopt" in discopt.__file__, discopt.__file__
from discopt._relax.convexity import rules as _rules  # noqa: E402

assert hasattr(_rules, "_const_scale_sign"), (
    f"loaded a discopt without the #944 array-constant curvature rule: {_rules.__file__}"
)


def _sq(expr):
    """``expr ** 2`` — convex, non-affine, for probing the scaling rules."""
    return expr * expr


# ──────────────────────────────────────────────────────────────────────
# Unit: the curvature rule itself
# ──────────────────────────────────────────────────────────────────────


class TestArrayConstantScaling:
    """``c ⊙ f`` and ``f / c`` for a constant ARRAY ``c``."""

    @staticmethod
    def _model():
        m = dm.Model("scale")
        x = m.continuous("x", shape=(3,), lb=-5, ub=5)
        return m, x

    def test_affine_survives_any_constant_array(self):
        """An affine operand stays affine under any constant array — even mixed."""
        checked = 0
        for c in (
            np.array([1.0, 2.0, 3.0]),  # all positive
            np.array([-1.0, -2.0, -3.0]),  # all negative
            np.array([1.0, -2.0, 3.0]),  # MIXED — still affine
            np.array([0.0, 0.0, 0.0]),  # all zero
            np.array([0.0, 2.0, 0.0]),  # zeros alongside positives
        ):
            m, x = self._model()
            for expr in (c * x, x * c):
                assert classify_expr(expr, m, {}) == Curvature.AFFINE, c
                checked += 1
        assert checked == 10, checked  # anti-vacuity (§6)

    def test_convex_operand_scales_by_uniform_sign(self):
        checked = 0
        cases = [
            (np.array([1.0, 2.0, 3.0]), Curvature.CONVEX),
            (np.array([0.0, 2.0, 0.0]), Curvature.CONVEX),  # zeros are harmless
            (np.array([-1.0, -2.0, -3.0]), Curvature.CONCAVE),
            (np.array([-1.0, 0.0, -3.0]), Curvature.CONCAVE),
            (np.array([0.0, 0.0, 0.0]), Curvature.AFFINE),  # 0 * anything
        ]
        for c, want in cases:
            m, x = self._model()
            for expr in (c * _sq(x), _sq(x) * c):
                assert classify_expr(expr, m, {}) == want, (c, want)
                checked += 1
        assert checked == 10, checked

    def test_mixed_sign_constant_refuses_on_nonaffine_operand(self):
        """No single scaling is valid, so the verdict must be UNKNOWN, not a guess.

        This is the soundness direction: claiming CONVEX here would let the
        solver certify a local optimum as global.
        """
        m, x = self._model()
        c = np.array([1.0, -2.0, 3.0])
        checked = 0
        for expr in (c * _sq(x), _sq(x) * c):
            assert classify_expr(expr, m, {}) == Curvature.UNKNOWN
            checked += 1
        assert checked == 2, checked

    def test_division_by_constant_array(self):
        checked = 0
        cases = [
            (np.array([1.0, 2.0, 4.0]), Curvature.CONVEX),
            (np.array([-1.0, -2.0, -4.0]), Curvature.CONCAVE),
            (np.array([1.0, -2.0, 4.0]), Curvature.UNKNOWN),  # mixed, non-affine
        ]
        for c, want in cases:
            m, x = self._model()
            assert classify_expr(_sq(x) / c, m, {}) == want, (c, want)
            checked += 1
        # An affine numerator divides entrywise to affine even for a mixed divisor.
        m, x = self._model()
        assert classify_expr(x / np.array([1.0, -2.0, 4.0]), m, {}) == Curvature.AFFINE
        checked += 1
        assert checked == 4, checked

    def test_division_by_array_with_a_zero_entry_refuses(self):
        """One zero entry makes the quotient undefined there — refuse, don't scale."""
        m, x = self._model()
        checked = 0
        for c in (np.array([1.0, 0.0, 4.0]), np.array([2.0, 1e-31, 4.0])):
            assert classify_expr(_sq(x) / c, m, {}) == Curvature.UNKNOWN, c
            checked += 1
        assert checked == 2, checked

    def test_non_finite_constant_refuses(self):
        """`inf`/`nan` coefficients refuse for EVERY operand curvature and arity.

        The affine arm is the one that matters. A first cut of this fix screened
        non-finite inside the sign helper, whose ``None`` then fell straight into
        the mixed-sign branch and returned AFFINE for `nan * affine` -- emitting a
        *proof* about a broken model, and (for an array constant) looser than the
        pre-fix walker, which had no array branch at all and answered UNKNOWN.
        Both directions are pinned here so neither can drift back.

        `_classify_division` has always refused non-finite divisors; these
        assertions are what keep the product rule agreeing with it.
        """
        checked = 0
        for c in (
            np.array([1.0, np.inf, 3.0]),
            np.array([1.0, np.nan, 3.0]),
            np.inf,
            -np.inf,
            np.nan,
        ):
            for operand_is_convex in (False, True):
                m, x = self._model()
                operand = _sq(x) if operand_is_convex else x
                assert classify_expr(c * operand, m, {}) == Curvature.UNKNOWN, (
                    c,
                    operand_is_convex,
                )
                assert classify_expr(operand / c, m, {}) == Curvature.UNKNOWN, (
                    c,
                    operand_is_convex,
                )
                checked += 2
        assert checked == 20, checked

    def test_scalar_behaviour_is_unchanged(self):
        """The pre-existing 0-d cases must classify exactly as before."""
        m, x = self._model()
        checked = 0
        for k, want in (
            (2.0, Curvature.CONVEX),
            (-2.0, Curvature.CONCAVE),
            (0.0, Curvature.AFFINE),
        ):
            assert classify_expr(k * _sq(x), m, {}) == want, k
            assert classify_expr(_sq(x) * k, m, {}) == want, k
            assert classify_expr(k * x, m, {}) == Curvature.AFFINE, k
            checked += 3
        assert classify_expr(_sq(x) / 2.0, m, {}) == Curvature.CONVEX
        assert classify_expr(_sq(x) / -2.0, m, {}) == Curvature.CONCAVE
        checked += 2
        assert checked == 11, checked


# ──────────────────────────────────────────────────────────────────────
# The Rust bridge: negative subscripts
# ──────────────────────────────────────────────────────────────────────


class TestNegativeIndexReachesTheRepr:
    """``model_to_repr`` must accept ``x[-1]`` and resolve it the way numpy does."""

    @pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
    def test_negative_index_resolves_like_numpy(self, shape):
        """Oracle is numpy's own indexing of an ``arange`` probe, not a reimplementation."""
        size = int(np.prod(shape))
        probe = np.arange(size, dtype=float).reshape(shape)
        checked = 0
        for flat in range(size):
            pos = np.unravel_index(flat, shape)
            neg = tuple(int(p) - int(s) for p, s in zip(pos, shape))
            want = float(probe[pos])
            for idx in (pos, neg):
                key = idx if len(shape) > 1 else idx[0]
                m = dm.Model("neg")
                x = m.continuous("x", shape=shape, lb=-100, ub=100)
                m.minimize(x[key] * 1.0)
                got = float(cached_evaluator(m).evaluate_objective(probe.ravel()))
                assert got == want, f"{shape}{key}: got {got}, want {want}"
                # The whole point of #944's second defect: this used to raise.
                model_to_repr(m, getattr(m, "_builder", None))
                checked += 1
        assert checked == 2 * size, checked  # anti-vacuity (§6)

    def test_out_of_range_negative_index_is_refused(self):
        """Refuse rather than wrap: a wrong-but-in-range slot names another variable.

        ``x[-4]`` on a shape-(3,) variable is already rejected by the ``__getitem__``
        guard (#816), so the bridge is exercised through the lazy
        ``IndexExpression`` constructor the guard deliberately leaves open — the
        node shape the GAMS/NL importers build.
        """
        from discopt.modeling.core import IndexExpression

        m = dm.Model("oob")
        x = m.continuous("x", shape=(3,), lb=0, ub=1)
        m.minimize(IndexExpression(x, -4) * 1.0)
        with pytest.raises(IndexError):
            model_to_repr(m, getattr(m, "_builder", None))

    def test_boolean_subscript_is_refused(self):
        """`x[True]` now raises instead of resolving as `x[1]` -- pin it deliberately.

        `bool` is a subclass of `int`, so the old `usize` extraction accepted
        `True` and named slot 1. numpy reads a bool as a *mask*, not an index, so
        that silently disagreed with the evaluator the relaxation is built against.
        Refusing is the intended behaviour, not an oversight -- this test exists so
        a future reader does not "restore" the old one.
        """
        from discopt.modeling.core import IndexExpression

        m = dm.Model("boolidx")
        x = m.continuous("x", shape=(3,), lb=0, ub=1)
        m.minimize(IndexExpression(x, True) * 1.0)
        with pytest.raises(IndexError):
            model_to_repr(m, getattr(m, "_builder", None))

    def test_classification_survives_a_negative_index(self):
        """A negative index must not demote a QP to ``NLP`` (the silent degradation)."""
        from discopt._relax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("qp_neg")
        x = m.continuous("x", shape=(3,), lb=-5, ub=5)
        m.subject_to(x[0] + x[1] + x[2] >= 1.0)
        m.minimize(x[-1] ** 2 + x[0] ** 2)
        assert classify_problem(m) == ProblemClass.QP


# ──────────────────────────────────────────────────────────────────────
# End to end: the model from the issue
# ──────────────────────────────────────────────────────────────────────


def _minimum_energy_model(T=2.0, u_bound=3.0, nfe=20, ncp=3):
    """The issue's model: min ∫₀ᵀ u² dt + 10·x(T)² s.t. ẋ = -x + u, x(0)=1."""
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("min_energy")
    cs = ContinuousSet("t", bounds=(0, T), nfe=nfe, ncp=ncp)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.add_control("u", bounds=(-u_bound, u_bound))
    dae.set_ode(lambda t, s, a, c: {"x": -s["x"] + c["u"]})
    dae.discretize()
    x_var = dae.get_state("x")
    m.minimize(dae.integral(lambda t, s, a, c: c["u"] ** 2) + 10.0 * x_var[-1, -1] ** 2)
    return m, dae


def _riccati_optimum(T: float, S_T: float) -> float:
    """Analytic optimal cost of the unconstrained LQR, independent of the solver.

    For ẋ = ax + bu (a = -1, b = 1) with cost ∫₀ᵀ R u² dt + S_T x(T)² and R = 1,
    the Riccati variable obeys Ṡ = -2aS + b²S²/R = S² + 2S with S(T) = S_T, whose
    separated solution gives ½·ln(S/(S+2)) = t + C. The optimal cost is
    S(0)·x(0)² = S(0). The control bound |u| ≤ 3 is inactive along this
    trajectory, so the box-constrained optimum coincides with it.
    """
    c = 0.5 * np.log(S_T / (S_T + 2.0)) - T
    r = np.exp(2.0 * c)  # = S0 / (S0 + 2)
    return float(2.0 * r / (1.0 - r))


class TestMinimumEnergyTerminates:
    def test_convexity_is_recognised(self):
        """The gate that was wrong: this model is convex and must be seen as such."""
        m, _ = _minimum_energy_model()
        is_convex, mask = classify_model(m, use_certificate=True)
        assert mask, "no constraints classified — the probe would be vacuous"
        assert all(mask), f"a linear collocation row classified nonconvex: {mask}"
        assert is_convex

    def test_solves_to_certified_optimality_without_branching(self):
        """0 nodes, certified optimal, and the objective matches the Riccati value.

        Before the fix this ran ~9k nodes without certifying and blew an 800 s
        pytest timeout. The ``time_limit`` here is what makes a regression report
        as a failed assertion rather than as a harness timeout (issue item 3).

        Deliberately NOT ``@pytest.mark.slow``. This is the one test that asserts
        the *routing* rather than the classification, so it is the one that would
        have caught #944 -- and every CI lane deselects ``slow``, which would have
        left the fix's actual claim unexercised there. It costs ~2 s now that the
        model takes the convex path, and the ``time_limit`` bounds the worst case
        at 120 s, so a regression fails an assertion instead of hanging the lane.
        """
        m, dae = _minimum_energy_model(T=2.0)
        result = m.solve(time_limit=120)

        assert result.status == "optimal", result.status
        assert result.gap_certified is True
        assert getattr(result, "convex_fast_path", False) is True
        # A convex model must not branch at all.
        assert (result.node_count or 0) == 0, result.node_count

        # Independent oracle: the analytic LQR cost. The gap is collocation
        # discretization error (20 elements, 3 Radau points), not solver error.
        analytic = _riccati_optimum(T=2.0, S_T=10.0)
        assert abs(result.objective - analytic) < 1e-3, (result.objective, analytic)

        # And the certificate invariant.
        assert result.bound <= result.objective + 1e-6

        _, x_vals = dae.extract_solution(result, "x")
        assert abs(x_vals[-1]) < 0.5

    @pytest.mark.slow
    def test_notebook_variant_also_terminates(self):
        """``docs/notebooks/tutorial_dae.ipynb`` cell 11: T = 5, |u| ≤ 2."""
        m, _ = _minimum_energy_model(T=5.0, u_bound=2.0)
        result = m.solve(time_limit=120)
        assert result.status == "optimal", result.status
        assert (result.node_count or 0) == 0, result.node_count
        analytic = _riccati_optimum(T=5.0, S_T=10.0)
        assert abs(result.objective - analytic) < 1e-3, (result.objective, analytic)
