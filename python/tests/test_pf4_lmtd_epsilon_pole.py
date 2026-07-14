"""PF4 falsification guard — the heatexch LMTD term has an in-box ε-pole.

The heatexch_gen* instances encode the Log-Mean Temperature Difference as the
literal term ``(a - b) / log(a / (eps + b))`` with ``eps = 1e-6`` and both
temperatures bounded ``[10, +inf)``. The PF4 spike proposed the classical
logarithmic-mean envelope ``GM(a,b) <= LMTD <= AM(a,b)`` — over-estimator
``w <= (a+b)/2`` (AM), under-estimator ``w >= chord(sqrt(a*b))`` (GM secant).

That envelope is SOUND for the ε-FREE mean ``(a-b)/log(a/b)`` (which the spike
sampled) but UNSOUND for the term the model actually contains: the ``+ eps`` in
the denominator does not remove the LMTD singularity, it MOVES it to the line
``a = eps + b``, which lies strictly inside the ``[10, +inf)^2`` box. There the
denominator ``log(a/(eps+b))`` crosses zero with a non-zero numerator, so
``w = (a-b)/log(a/(eps+b))`` is genuinely unbounded (``+-inf``). Any finite
over-estimator therefore CUTS feasible points near that line — the worst class of
bug (false-infeasible / lost optimum, CLAUDE.md §Development-Philosophy).

Consequence (measured, this file): the current relaxation, which places NO finite
envelope on the ratio aux over a pole-straddling box (the sound interval floor is
``[-inf, +inf]``), is CORRECT — "unbounded" is the sound answer, not a fixable
hole. These tests pin the falsification so a future ungated AM/GM envelope cannot
be re-introduced without tripping them.

See ``docs/dev/pf4-rootgap-spike.md`` §6 (falsification) and
``docs/dev/sota-proof-plan.md`` §3 PF4 row (KILL — soundness).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model

_EPS = 1e-6


def _lmtd_eps(a, b):
    """The EXACT model term (a-b)/log(a/(eps+b)) — the value of the ratio aux."""
    return (a - b) / np.log(a / (_EPS + b))


@pytest.mark.unit
def test_epsilon_term_pole_is_inside_the_heatexch_box():
    """The denominator zero (pole) a = eps + b lies inside the [10, +inf)^2 box."""
    b = 10.0
    a_pole = _EPS + b  # = 10.000001, well inside a in [10, +inf)
    assert 10.0 <= a_pole
    # Approaching the pole from the numerator side the term explodes toward +inf:
    # AM(a,b) ~ 10 here, yet w grows without bound as the offset -> 0.
    assert _lmtd_eps(a_pole + 1e-9, b) > 1e3  # already ~1e4, i.e. ~1000x AM
    assert _lmtd_eps(a_pole + 1e-11, b) > 1e5  # closer still: ~1e6, unbounded


@pytest.mark.unit
def test_am_overestimator_is_unsound_for_the_epsilon_term():
    """AM ``w <= (a+b)/2`` is violated by the exact ε-term at feasible points.

    This is the guard: if a future change makes AM appear sound here (e.g. by
    silently switching to the ε-free mean, or narrowing the box away from the
    pole), the assumption changed and this test must be revisited — never weaken
    it to make an envelope pass.
    """
    rng = np.random.default_rng(0)
    n = 2_000_000
    a = rng.uniform(10.0, 650.0, n)
    b = rng.uniform(10.0, 650.0, n)
    w = _lmtd_eps(a, b)
    am = 0.5 * (a + b)
    violations = int(np.count_nonzero(w - am > 1e-6))
    assert violations > 0, "AM over-estimator must be UNSOUND for the ε-LMTD term"
    # And the violation magnitude is not a rounding whisker — the term overshoots
    # AM by O(1) near the pole line a = eps + b.
    assert float(np.max(w - am)) > 1.0


@pytest.mark.unit
def test_gm_underestimator_is_unsound_for_the_epsilon_term():
    """GM ``w >= sqrt(a*b)`` is violated (w -> -inf on the other pole side)."""
    rng = np.random.default_rng(1)
    n = 2_000_000
    a = rng.uniform(10.0, 650.0, n)
    b = rng.uniform(10.0, 650.0, n)
    w = _lmtd_eps(a, b)
    gm = np.sqrt(a * b)
    violations = int(np.count_nonzero(gm - w > 1e-6))
    assert violations > 0, "GM under-estimator must be UNSOUND for the ε-LMTD term"
    assert float(np.max(gm - w)) > 1.0


@pytest.mark.unit
def test_sign_definite_denominator_gate_is_insufficient_for_am():
    """Even a strictly sign-definite denominator (a > eps + b throughout) does NOT
    make AM sound: near the pole, ``(a-b)/log(a/(eps+b))`` still exceeds ``(a+b)/2``.

    This kills the tempting "just require the denominator sign-definite" fix — the
    margin needed for AM validity is quantitative and box-dependent, and it fails
    exactly at the small-approach (pinch) region where good HEN solutions live.
    """
    rng = np.random.default_rng(2)
    n = 3_000_000
    b = rng.uniform(10.0, 20.0, n)
    a = b + rng.uniform(2e-6, 1.0, n)  # a > eps + b everywhere => log(a/(eps+b)) > 0
    assert np.all(a > _EPS + b)
    w = _lmtd_eps(a, b)
    am = 0.5 * (a + b)
    assert int(np.count_nonzero(w - am > 1e-6)) > 0


def _build_tracked_relaxation(model, box):
    """Build the uniform relaxation with aux->exact-expr tracking (soundness harness)."""
    from discopt._jax import uniform_relax as ur
    from discopt._jax.canonical_expr import canonicalize

    flat_lb = np.asarray(box[0], dtype=np.float64)
    flat_ub = np.asarray(box[1], dtype=np.float64)
    dag = canonicalize(model)
    ctx = ur._Builder(model, flat_lb, flat_ub, track_aux_exprs=True)
    roots = ([dag.objective] if dag.objective is not None else []) + list(dag.constraints)
    for r in roots:
        ctx.rep(r)
    return ctx


def _eval_expr(model, expr, xv):
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_eval import evaluate_interval

    box = {}
    off = 0
    for v in model._variables:
        size = int(v.size)
        shape = tuple(getattr(v, "shape", ()) or ())
        pt = np.asarray(xv[off : off + size], dtype=np.float64).reshape(shape)
        box[v] = Interval(pt, pt)
        off += size
    return float(np.asarray(evaluate_interval(expr, model, box).lo))


@pytest.mark.unit
def test_current_relaxation_does_not_cut_near_pole_feasible_points():
    """The shipped relaxation of the LMTD term emits NO row that cuts a feasible
    near-pole point — i.e. it correctly withholds the (unsound) AM/GM envelope.

    If someone adds an ungated LMTD over-/under-estimator, the near-pole lifted
    points below will be cut and this test fails — which is the whole point.
    """
    m = Model()
    a = m.continuous("a", lb=10.0, ub=650.0)
    b = m.continuous("b", lb=10.0, ub=650.0)
    # Minimize -w so the LP "wants" w large (the heatexch area-cost direction).
    w = (a - b) / dm.log(a / (_EPS + b))
    m.minimize(-w)

    box = (np.array([10.0, 10.0]), np.array([650.0, 650.0]))
    ctx = _build_tracked_relaxation(m, box)

    # Feasible points hugging the pole line a = eps + b (both sides) where w blows up.
    probes = [
        (10.0 + 2e-6, 10.0),
        (10.0 + 1.1e-6, 10.0),
        (10.0 + 0.9e-6, 10.0),
        (494.52 + 5e-5, 494.52),
        (100.0, 100.0),  # a == b exactly: w == 0 (ε breaks the a=b limit)
    ]
    n_cols = len(ctx.col_lb)
    for av, bv in probes:
        xv = np.array([av, bv], dtype=np.float64)
        z = np.zeros(n_cols)
        z[: ctx.n_orig] = xv
        for j in sorted(ctx.aux_expr):
            z[j] = _eval_expr(m, ctx.aux_expr[j], xv)
        for coeffs, rhs in ctx.rows:
            lhs = sum(c * z[jj] for jj, c in coeffs.items())
            assert lhs <= rhs + 1e-6, (
                f"UNSOUND: feasible near-pole point (a={av}, b={bv}) cut: {lhs} > {rhs}"
            )
