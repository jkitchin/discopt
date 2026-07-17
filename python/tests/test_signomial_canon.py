"""Signomial canonicaliser + rigorous box lower bound (issue #114).

Covers:

* Canonicaliser round-trip: an extracted :class:`SignomialForm` re-evaluates to
  the same value as the model expression at sampled points; like terms merge.
* Recognition boundary: non-signomial objectives and non-positive variables are
  refused (``None``).
* Soundness (the hard requirement): the interval lower bound is ``<=`` the true
  global minimum on small signomials with brute-forced optima, for both
  minimisation-relevant and unbounded-box cases.
* Negative controls: genuinely non-convex mixed-sign signomials are NEVER
  promoted to a convex / global-optimal verdict — ``certified`` stays ``False``,
  and the GP path keeps abstaining (``is_log_convex`` False, ``classify_gp`` /
  ``solve_gp`` None).
* DC-envelope bridge: the ``(sigma, log_c, exps)`` terms feed the certified
  log-domain envelope and reproduce ``cv <= s <= cc``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from discopt._jax.convexity.signomial import (
    SignomialForm,
    is_signomial,
    signomial_box_lower_bound,
    signomial_dc_terms,
    signomial_relaxation,
)
from discopt._jax.symbolic.signed_signomial import signed_signomial_dc_envelope
from discopt.gp import classify_gp, is_log_convex, solve_gp
from discopt.modeling.core import Model

pytestmark = pytest.mark.relaxation


def _brute_min(form: SignomialForm, box: dict[int, tuple[float, float]], n: int = 121) -> float:
    """Dense-grid global minimum of the signomial over the (finite) box."""
    offsets = sorted(box)
    grids = [np.linspace(box[o][0], box[o][1], n) for o in offsets]
    best = math.inf
    for pt in itertools.product(*grids):
        x_by_off = {o: pt[j] for j, o in enumerate(offsets)}
        best = min(best, form.evaluate(x_by_off))
    return best


# ──────────────────────────────────────────────────────────────────────
# Canonicaliser
# ──────────────────────────────────────────────────────────────────────


class TestCanonicaliser:
    def test_roundtrip_mixed_sign(self):
        m = Model("sig")
        x = m.continuous("x", lb=0.2, ub=3.0)
        y = m.continuous("y", lb=0.3, ub=2.5)
        expr = x * y - 2.0 * x / y + 0.5 * y**2
        form = is_signomial(expr, m)
        assert form is not None
        assert form.is_mixed_sign
        assert form.has_negative_term
        rng = np.random.default_rng(0)
        for _ in range(50):
            xv = float(rng.uniform(0.2, 3.0))
            yv = float(rng.uniform(0.3, 2.5))
            expected = xv * yv - 2.0 * xv / yv + 0.5 * yv**2
            assert form.evaluate({0: xv, 1: yv}) == pytest.approx(expected, rel=1e-12)

    def test_like_terms_merge_and_cancel(self):
        m = Model("merge")
        x = m.continuous("x", lb=0.1, ub=5.0)
        # 3x - x = 2x (merges to one term); x - x cancels to nothing.
        form = is_signomial(3.0 * x - x, m)
        assert form is not None
        assert len(form.monomials) == 1
        assert form.monomials[0].coeff == pytest.approx(2.0)
        assert not form.has_negative_term  # 2x is a lone positive term
        assert is_signomial(x - x, m) is None  # full cancellation -> not a signomial

    def test_posynomial_is_signomial_but_not_mixed(self):
        m = Model("posy")
        x = m.continuous("x", lb=0.1, ub=5.0)
        y = m.continuous("y", lb=0.1, ub=5.0)
        form = is_signomial(x * y + x / y, m)
        assert form is not None
        assert not form.is_mixed_sign
        assert not form.has_negative_term

    def test_rejects_nonpositive_variable(self):
        m = Model("neg_lb")
        x = m.continuous("x", lb=-1.0, ub=5.0)  # lb <= 0: no log domain
        assert is_signomial(x * x - x, m) is None

    def test_rejects_non_monomial_term(self):
        m = Model("nonmono")
        x = m.continuous("x", lb=0.1, ub=5.0)
        # exp(x) is not a monomial -> not a signomial.
        import discopt.modeling as dm

        assert is_signomial(dm.exp(x) - x, m) is None


# ──────────────────────────────────────────────────────────────────────
# Rigorous lower bound (soundness)
# ──────────────────────────────────────────────────────────────────────


class TestLowerBoundSoundness:
    def test_bound_below_true_min_1d(self):
        # S = x^2 - 3x on [0.2, 3]; true min = -2.25 at x = 1.5.
        m = Model("s1")
        x = m.continuous("x", lb=0.2, ub=3.0)
        form = is_signomial(x**2 - 3.0 * x, m)
        assert form is not None
        lb = signomial_box_lower_bound(form, m)
        true_min = _brute_min(form, {0: (0.2, 3.0)})
        assert math.isfinite(lb)
        assert lb <= true_min + 1e-9  # rigorous
        assert lb > -1e6  # non-trivial (finite, not garbage)

    def test_bound_below_true_min_2d(self):
        # S = sqrt(x*y) - 1/x - 1/y on [0.3, 3]^2 (this bound is tight here).
        m = Model("s2")
        x = m.continuous("x", lb=0.3, ub=3.0)
        y = m.continuous("y", lb=0.3, ub=3.0)
        import discopt.modeling as dm

        form = is_signomial(dm.sqrt(x * y) - 1.0 / x - 1.0 / y, m)
        assert form is not None
        assert form.is_mixed_sign
        lb = signomial_box_lower_bound(form, m)
        true_min = _brute_min(form, {0: (0.3, 3.0), 1: (0.3, 3.0)})
        assert lb <= true_min + 1e-9

    def test_bound_below_true_min_random_suite(self):
        """Many random mixed-sign signomials: LB <= grid min, always."""
        rng = np.random.default_rng(7)
        for _ in range(40):
            m = Model("rand")
            x = m.continuous("x", lb=0.4, ub=2.5)
            y = m.continuous("y", lb=0.4, ub=2.5)
            ax, ay = rng.integers(-2, 3, size=2)
            bx, by = rng.integers(-2, 3, size=2)
            cpos = float(rng.uniform(0.5, 3.0))
            cneg = float(rng.uniform(0.5, 3.0))
            expr = cpos * x ** int(ax) * y ** int(ay) - cneg * x ** int(bx) * y ** int(by)
            form = is_signomial(expr, m)
            if form is None or not form.monomials:
                continue
            lb = signomial_box_lower_bound(form, m)
            true_min = _brute_min(form, {0: (0.4, 2.5), 1: (0.4, 2.5)}, n=81)
            assert lb <= true_min + 1e-6, (ax, ay, bx, by, cpos, cneg, lb, true_min)

    def test_open_box_negative_term_gives_minus_inf(self):
        # Subtracted term with positive exponent and ub = inf -> unbounded below.
        m = Model("open")
        x = m.continuous("x", lb=0.5, ub=np.inf)
        form = is_signomial(1.0 / x - x, m)
        assert form is not None
        lb = signomial_box_lower_bound(form, m)
        assert lb == -math.inf  # honest, sound "no useful bound"

    def test_open_box_bounded_case_is_finite(self):
        # Only a decaying negative term as x -> inf: bound stays finite.
        m = Model("open2")
        x = m.continuous("x", lb=0.5, ub=np.inf)
        form = is_signomial(x - 1.0 / x, m)  # -1/x -> 0 as x->inf, +x min at 0.5
        assert form is not None
        lb = signomial_box_lower_bound(form, m)
        assert math.isfinite(lb)
        # true inf over [0.5, inf): at x=0.5, 0.5 - 2 = -1.5; bound must be <=.
        assert lb <= -1.5 + 1e-9


# ──────────────────────────────────────────────────────────────────────
# Negative controls: never a false certificate; GP abstention preserved
# ──────────────────────────────────────────────────────────────────────


class TestNegativeControls:
    def _nonconvex_model(self):
        # S = x - x^2 (concave, unbounded-below flavour) type non-convex signomial.
        m = Model("nc")
        x = m.continuous("x", lb=0.2, ub=4.0)
        y = m.continuous("y", lb=0.2, ub=4.0)
        # A genuinely non-convex signomial: negative bilinear + positive squares.
        m.minimize(x**2 + y**2 - 4.0 * x * y)
        return m

    def test_relaxation_never_certifies(self):
        m = self._nonconvex_model()
        relax = signomial_relaxation(m)
        assert relax is not None
        assert relax.is_mixed_sign
        assert relax.certified is False  # hard gate: never certified
        # The reported bound is a valid lower bound on the true global min.
        form = relax.form
        true_min = _brute_min(form, {0: (0.2, 4.0), 1: (0.2, 4.0)})
        assert relax.lower_bound <= true_min + 1e-6

    def test_gp_path_still_abstains_on_signomial(self):
        m = self._nonconvex_model()
        # The GP recogniser must NOT touch a mixed-sign objective.
        assert is_log_convex(m) is False
        assert classify_gp(m) is None
        assert solve_gp(m) is None

    def test_maximisation_not_relaxed(self):
        m = Model("maxi")
        x = m.continuous("x", lb=0.2, ub=4.0)
        m.maximize(x - x**2)
        # signomial_relaxation only handles minimisation; must return None
        # (no bound in the wrong direction).
        assert signomial_relaxation(m) is None

    def test_many_nonconvex_never_certified_and_bound_sound(self):
        rng = np.random.default_rng(11)
        for _ in range(30):
            m = Model("ncr")
            x = m.continuous("x", lb=0.3, ub=3.0)
            y = m.continuous("y", lb=0.3, ub=3.0)
            expr = (
                float(rng.uniform(0.5, 2.0)) * x ** int(rng.integers(1, 3))
                - float(rng.uniform(0.5, 2.0)) * x * y
                + float(rng.uniform(0.5, 2.0)) * y ** int(rng.integers(1, 3))
            )
            m.minimize(expr)
            relax = signomial_relaxation(m)
            if relax is None:
                continue
            assert relax.certified is False
            assert is_log_convex(m) is False
            true_min = _brute_min(relax.form, {0: (0.3, 3.0), 1: (0.3, 3.0)}, n=81)
            assert relax.lower_bound <= true_min + 1e-6


# ──────────────────────────────────────────────────────────────────────
# Bridge to the certified DC envelope
# ──────────────────────────────────────────────────────────────────────


class TestDCBridge:
    def test_dc_terms_reproduce_envelope_bounds(self):
        m = Model("bridge")
        x = m.continuous("x", lb=0.3, ub=4.0)
        y = m.continuous("y", lb=0.3, ub=4.0)
        import discopt.modeling as dm

        form = is_signomial(x * y - 2.0 * dm.sqrt(x) / y + 1.5 / (x * y), m)
        assert form is not None
        terms, offsets = signomial_dc_terms(form)
        assert offsets == [0, 1]
        u_lb = np.array([math.log(0.3), math.log(0.3)])
        u_ub = np.array([math.log(4.0), math.log(4.0)])
        rng = np.random.default_rng(3)
        for _ in range(60):
            u = u_lb + rng.uniform(size=2) * (u_ub - u_lb)
            cv, cc = signed_signomial_dc_envelope(u, terms, u_lb, u_ub)
            # Reference: evaluate the signomial directly at x = exp(u).
            x_by_off = {0: math.exp(u[0]), 1: math.exp(u[1])}
            s = form.evaluate(x_by_off)
            assert float(cv) <= s + 1e-9
            assert float(cc) >= s - 1e-9
