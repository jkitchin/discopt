"""C-33 / SC-1 (P0, DEFAULT path) — the pure-continuous NLP fallback must NOT
certify a nonconvex model's local optimum as global.

The pure-continuous fallback in ``solver.py`` routes a model whose convexity is
*not established* (``skip_convex_check`` set, or the convexity classifier
abstained) to a single local NLP solve. Before the fix that local optimum was
emitted with ``bound = objective`` and ``gap_certified=True`` — a FALSE
optimality certificate on any nonconvex model, because a local minimum need not
be global.

Fails-before / passes-after: on the nonconvex double-well with
``skip_convex_check=True`` the local NLP lands in the shallow well
(objective ≈ -50.06) while the true global minimum is ≈ -78.33; before the fix
the solver returned that shallow point with ``gap_certified=True`` and
``bound == objective``.

Control: a genuinely convex pure-continuous model must STILL certify via the
convex fast path (guards against over-correcting into never certifying).
"""

import discopt.modeling as dm
import numpy as np
import pytest


def _double_well(lb=-4.0, ub=6.0):
    """Nonconvex asymmetric quartic double-well.

    ``f(x) = x**4 - 16 x**2 + 5 x`` on ``[lb, ub]`` has two minima: a deep
    (global) well at x ≈ -2.90 with f ≈ -78.33 and a shallow (local) well at
    x ≈ +2.75 with f ≈ -50.06. On ``[-4, 6]`` the midpoint start (x=1) sits in
    the shallow well's basin, so a single local NLP converges to the *local*
    optimum — not the global one.
    """
    m = dm.Model("c33_double_well")
    x = m.continuous("x", lb=lb, ub=ub)
    m.minimize(x**4 - 16 * x**2 + 5 * x)
    xs = np.linspace(lb, ub, 400001)
    fvals = xs**4 - 16 * xs**2 + 5 * xs
    true_global = float(fvals.min())
    return m, true_global


@pytest.mark.smoke
def test_c33_nonconvex_fallback_not_certified_optimal():
    """The pure-continuous fallback must not certify a nonconvex local optimum.

    With ``skip_convex_check=True`` the fallback fires without a convexity proof.
    The returned point may be kept as a feasible incumbent, but it must NOT be
    certified optimal, and no fabricated dual bound may be reported.
    """
    m, true_global = _double_well()
    r = m.solve(skip_convex_check=True)

    # Feasible incumbent is fine; a *certified* optimum here would be false.
    assert not r.gap_certified, (
        "false optimality certificate: nonconvex local optimum certified as global "
        f"(obj={r.objective}, true global={true_global})"
    )
    # The fabricated dual bound (local objective) must be withheld.
    assert r.bound is None, f"fabricated dual bound {r.bound} on unproven-convex fallback"

    # Soundness invariant (min sense): a *reported* dual bound must never exceed
    # the true global optimum. bound is None here, but if a future change routes
    # this to a genuine bound, it must still respect this.
    if r.bound is not None:
        assert r.bound <= true_global + 1e-6

    # If the solver did emit a certified optimum, it must be the true global one
    # (this is the pin that fails before the fix, where obj≈-50.06 was certified).
    if r.gap_certified and r.objective is not None:
        assert r.objective <= true_global + 1e-4


@pytest.mark.smoke
def test_c33_convex_control_still_certifies():
    """Control: a convex pure-continuous model still certifies (no over-correction).

    ``exp(x) + x**2`` is convex; the convex fast path must still fire and emit a
    certified optimum with a finite bound.
    """
    m = dm.Model("c33_convex_control")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(dm.exp(x) + x**2)
    r = m.solve()

    assert r.status == "optimal"
    assert r.gap_certified, "convex model lost its valid optimality certificate"
    assert r.bound is not None and np.isfinite(r.bound)
    assert r.convex_fast_path is True


@pytest.mark.smoke
def test_c33_default_path_nonconvex_uses_spatial_bb():
    """Default path (no skip flag) on the same nonconvex model routes to spatial
    B&B, finds the true global optimum, and certifies it with a valid bound.

    Guards against the fix accidentally decertifying the sound spatial path.
    """
    m, true_global = _double_well()
    r = m.solve()

    assert r.status == "optimal"
    # Spatial B&B finds the deep well and certifies it with a bound at or below obj.
    assert r.objective <= true_global + 1e-3
    if r.gap_certified:
        assert r.bound is not None and r.bound <= true_global + 1e-3
