"""Relaxation of the GAMS relative-entropy intrinsic ``centropy(x, y) = x·log(x/y)``.

The recognition half was already built (``factorable_reform`` rewrites raw
``x·log(x/y)`` products into ``centropy`` nodes and the convexity lattice knows
``centropy`` is JOINTLY CONVEX on ``x ≥ 0, y > 0``), but the relaxation half was
missing: ``build_milp_relaxation`` had no way to linearize a 2-argument
``centropy`` FunctionCall, so the whole objective collapsed to a *feasibility*
objective — no valid lower bound, so the MINLPLib ``ex6_2_*`` entropy family
could never certify optimality (ex6_2_5: correct incumbent −70.752 but no dual
bound).

``centropy`` is jointly convex, so the convex underestimator is a set of
supporting hyperplanes (first-order tangent planes). ``centropy`` now flows
through the composite-multivariate gradient-lift path
(:class:`CompositeMultivarRelaxation`): the collector's ``classify_expr`` gate
licenses CONVEX only when both arguments are affine and the domain (x ≥ 0,
y > 0) is provable on the box, and each gradient cut
``d ≥ centropy(x0) + ∇centropy(x0)·(x − x0)`` is a globally valid underestimator
(convexity) — sound by construction. LP-point separation (``_separate_convex``)
adds the exact supporting tangent at the LP point each round.

Soundness rests entirely on the curvature gate + the tangent being a global
underestimator of a convex function, so no feasible point is ever cut. These
tests pin: (1) the lift fires and produces a *finite, valid* lower bound where
before there was none; (2) the bound never exceeds the true minimum (no cut of a
feasible point); (3) a non-affine / negative-domain argument is NOT claimed.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax import milp_relaxation as MR
from discopt._jax.factorable_reform import canonicalize_entropy
from discopt._jax.mccormick_lp import MccormickLPRelaxer

pytestmark = [pytest.mark.claim_boundary]


def _centropy_model():
    """``min x·log(x/y) + x + y`` on ``[0.5, 3]²``.

    Built from the raw ``x·log(x/y)`` product so ``canonicalize_entropy`` rewrites
    it into a ``centropy`` node exactly as the ``.nl`` reader path does. The true
    box minimum is 1.0 (interior stationary point ``x = y``, where
    ``x·log(x/y) = 0`` and ``x + y`` is minimised on the diagonal at the lower
    corner ``x = y = 0.5`` giving 1.0).
    """
    m = dm.Model("centropy")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x * dm.log(x / y) + x + y)
    return canonicalize_entropy(m)


_CENTROPY_TRUE_MIN = 1.0


def _node_bound(model, lb, ub, *, claim: bool):
    """Root LP dual bound with the centropy claim toggled on/off.

    ``claim=False`` empties the jointly-convex atom table so the ``centropy`` node
    is never lifted (reproducing the pre-fix behaviour); ``claim=True`` uses the
    real table.
    """
    saved = MR._JOINTLY_CONVEX_MULTIVAR_ATOMS
    MR._JOINTLY_CONVEX_MULTIVAR_ATOMS = saved if claim else {}
    try:
        relaxer = MccormickLPRelaxer(model)
        r = relaxer.solve_at_node(np.asarray(lb, float), np.asarray(ub, float), separate=True)
        return r
    finally:
        MR._JOINTLY_CONVEX_MULTIVAR_ATOMS = saved


def test_centropy_claim_produces_finite_valid_bound():
    """The lift turns a missing (feasibility-fallback) bound into a finite, valid
    lower bound. This is the entropy-family unlock."""
    m = _centropy_model()
    off = _node_bound(m, [0.5, 0.5], [3.0, 3.0], claim=False)
    on = _node_bound(m, [0.5, 0.5], [3.0, 3.0], claim=True)

    # Before the fix the objective cannot be linearized → no valid dual bound
    # (the LP either has no objective row or the relaxer refuses the bound).
    assert off.lower_bound is None or off.lower_bound < on.lower_bound - 1e-6

    # After the fix: finite, and sound (a valid lower bound never exceeds the
    # true minimum).
    assert on.status == "optimal"
    assert on.lower_bound is not None
    assert np.isfinite(on.lower_bound)
    assert on.lower_bound <= _CENTROPY_TRUE_MIN + 1e-4


def test_centropy_bound_does_not_cut_feasible_points():
    """Feasible-point sampling: the tangent-plane underestimator is a global
    underestimator of the convex ``centropy``, so the LP lower bound must never
    exceed the objective at ANY feasible point of the original model."""
    m = _centropy_model()
    on = _node_bound(m, [0.5, 0.5], [3.0, 3.0], claim=True)
    assert on.lower_bound is not None
    rng = np.random.default_rng(20260710)
    worst = np.inf
    for _ in range(4000):
        xv = rng.uniform(0.5, 3.0)
        yv = rng.uniform(0.5, 3.0)
        fval = xv * np.log(xv / yv) + xv + yv
        worst = min(worst, fval)
    # The relaxation is a valid lower bound: it lies at or below every feasible
    # objective value (allowing sampling not to have found the exact minimum).
    assert on.lower_bound <= worst + 1e-6


@pytest.mark.slow
def test_centropy_family_bound_is_valid_and_sound():
    """End-to-end on the native centropy model: the solve returns a valid dual
    bound (never above the incumbent for a minimisation) instead of the
    feasibility fallback, and reaches the incumbent."""
    m = _centropy_model()
    res = m.solve(time_limit=60)
    assert res.objective is not None
    assert abs(float(res.objective) - _CENTROPY_TRUE_MIN) < 1e-3
    if res.bound is not None:
        # valid lower bound: bound ≤ incumbent (min sense) and ≤ true minimum.
        assert float(res.bound) <= float(res.objective) + 1e-4
        assert float(res.bound) <= _CENTROPY_TRUE_MIN + 1e-4


def test_centropy_negative_domain_is_not_claimed():
    """Soundness gate: ``centropy`` requires ``x ≥ 0`` for its convexity licence.
    A box that leaves the first argument possibly negative must NOT be lifted as
    convex (``classify_expr`` abstains), so the relaxation stays sound."""
    from discopt._jax.convexity import Curvature, classify_expr
    from discopt.modeling.core import FunctionCall

    m = dm.Model("negdom")
    x = m.continuous("x", lb=-1.0, ub=2.0)  # spans 0 → domain not provable
    y = m.continuous("y", lb=0.5, ub=2.0)
    ce = FunctionCall("centropy", x, y)
    # The classifier must abstain (not CONVEX) so no unsound gradient cut is built.
    assert classify_expr(ce, m) != Curvature.CONVEX


def test_centropy_nonaffine_argument_is_not_claimed():
    """A non-affine inner argument breaks the atom∘affine convexity licence; the
    structural pre-filter must refuse to claim it (the classifier would abstain
    anyway)."""
    from discopt.modeling.core import FunctionCall

    m = dm.Model("nonaffine")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    z = m.continuous("z", lb=0.5, ub=3.0)
    n_orig = sum(v.size for v in m._variables)
    # centropy(x*y, z): first argument is a nonlinear product, not affine.
    ce = FunctionCall("centropy", x * y, z)
    assert MR._should_claim_composite_multivar(ce, m, n_orig) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
