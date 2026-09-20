"""The certificate must survive scaling the objective (#1397, the CI probe).

The audit behind #1397 found four places where an absolute constant was compared
against a quantity carrying the problem's scale. Each was found by sweeping one
*internal* quantity and grading one *internal* verdict, which is the right way to
pin a known defect but the wrong way to catch the next one: a new site is found
only if someone already suspects it.

This is the class-level probe, and it is deliberately end-to-end and
implementation-blind. It asserts the one property every member of the class
violates, whatever the mechanism:

    a global solve's certificate must not depend on the objective's scale.

Multiplying a minimized objective by ``s > 0`` multiplies the optimum by ``s``
and leaves the argmin untouched, so the scaled problem's answer is known exactly
from the unscaled one. Every Hessian norm, gradient, objective value and
relaxation gap grows by ``s`` while every absolute tolerance stays put — so any
comparison that is dimensionally incoherent goes wrong somewhere along the sweep,
and a certificate that was sound at ``s = 1`` is no longer sound at ``s = 1e12``.
The two invariants checked are the ones that define the certificate:

* the dual bound never exceeds the true optimum (a false *lower* bound), and
* the incumbent never sits below it (an infeasible point reported as feasible).

Both are checked against optima known in closed form, not against a reference
run, so this probe cannot drift with the solver.

**The tolerance here is itself relative**, floored at 1. An absolute tolerance in
the probe would reproduce the very defect under audit: at ``s = 1e12`` a fixed
``1e-6`` slack is 18 orders of magnitude below the quantities being compared, and
the probe would pass by being unable to see anything.

Models are nonconvex and drawn to hit different relaxation routes -- indefinite
quadratic, bilinear/McCormick, univariate cubic, multilinear -- because the class
is about the *comparison*, not about any one route.

What this probe does NOT do, stated plainly so nobody reads more into a green run
--------------------------------------------------------------------------------
**It passes on the pre-fix tree.** Measured: 26 passed against `main` with the
four #1397 fixes absent (load gate asserted the marker symbol missing). It is a
forward-looking guard on the invariant, not a reproducer of those four defects,
and the arithmetic says it cannot be one:

* a false-PSD verdict licenses a bound whose error is at most
  ``|λ_min|·diam²/2`` with ``|λ_min| ≲ 5·u·‖H‖``, while the objective's own
  magnitude over the same box is ``~‖H‖·diam²/2`` -- so the *relative* error is
  ``~5u ≈ 1.1e-15``;
* ``rigorous_alpha``'s shortfall was ``O(u·‖A‖)`` for the same reason, hence
  ``O(u)`` relative at every scale.

A defect in this class therefore produces a certificate that is **provably**
invalid but invalid by a *relative* roundoff, which no relative end-to-end check
can resolve. That is not a reason to tolerate it (CLAUDE.md §1: a certificate
invalid by 1e-15 relative is still not a certificate, the error compounds over
thousands of nodes, and it flips prune decisions at a tie) -- it is the reason the
real probes for this class are the per-site parameter sweeps over the *internal*
quantity, each of which does fail before its fix and pass after:

* ``test_1397_scale_aware_psd_gate.py`` -- the convex-objective PSD gate,
* ``test_1397_scale_aware_convexity_slack.py`` -- the convexity certificate,
* ``test_1397_alphabb_alpha_dominates_nonconvexity.py`` -- both alphaBB alphas.

This file complements them by catching the coarser failure they cannot see: a
scale bug whose footprint *is* visible in the answer -- a mis-set tolerance, a
missing normalization, a relaxation that stops being valid outright -- anywhere
on the default solve path, including code that did not exist when the audit ran.
"""

from __future__ import annotations

import math

import pytest
from discopt import Model

#: Objective multipliers. ``1e0`` is the control: it must pass trivially, and if
#: it ever fails the probe is broken rather than the solver.
SCALES = (1e0, 1e3, 1e6, 1e9, 1e12)

#: Relative slack for the comparison, floored at 1 so the ``s = 1`` control is
#: byte-comparable with the historic absolute tolerance.
REL_TOL = 1e-6


def _indefinite_quadratic(scale):
    """``s·(x² − y²)`` over ``[−1, 1]²``. Hessian ``s·diag(2, −2)``: indefinite."""
    m = Model("indefinite_quadratic")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(scale * (x**2 - y**2))
    return m, -1.0 * scale


def _bilinear(scale):
    """``s·(−x·y)`` over ``[0, 1]²``. The McCormick route; optimum at ``(1, 1)``."""
    m = Model("bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(scale * (-x * y))
    return m, -1.0 * scale


def _cross_term_quadratic(scale):
    """``s·(x² + y² − 4xy)`` over ``[−1, 1]²``; eigenvalues ``s·(6, −2)``.

    On the diagonal ``y = x`` the form is ``−2x²``, minimized at ``|x| = 1``, and
    the box minimum of a quadratic with an indefinite Hessian is attained on the
    boundary — so the optimum is ``−2s``.
    """
    m = Model("cross_term_quadratic")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(scale * (x**2 + y**2 - 4.0 * x * y))
    return m, -2.0 * scale


def _univariate_cubic(scale):
    """``s·(x³ − 3x)`` over ``[−2, 2]``. Optimum ``−2s``, attained twice.

    ``f' = 3x² − 3`` vanishes at ``x = ±1``; ``f(1) = −2`` and ``f(−2) = −2`` tie,
    so the global minimum is degenerate — which is the interesting case for a
    branch-and-bound certificate, since neither optimum can be pruned by the
    other.
    """
    m = Model("univariate_cubic")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(scale * (x**3 - 3.0 * x))
    return m, -2.0 * scale


def _multilinear(scale):
    """``s·(xy + yz + xz)`` over ``[−1, 1]³``. Optimum ``−s``.

    For ``±1`` vectors ``(Σx)² = 3 + 2f``, so ``f = ((Σx)² − 3)/2`` takes only
    ``3`` and ``−1``; the form is multilinear, so its box minimum is at a vertex.
    """
    m = Model("multilinear")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    z = m.continuous("z", lb=-1.0, ub=1.0)
    m.minimize(scale * (x * y + y * z + x * z))
    return m, -1.0 * scale


#: ``(name, builder)``. Each builder returns ``(model, true_optimum_at_scale)``.
MODELS = (
    ("indefinite_quadratic", _indefinite_quadratic),
    ("bilinear", _bilinear),
    ("cross_term_quadratic", _cross_term_quadratic),
    ("univariate_cubic", _univariate_cubic),
    ("multilinear", _multilinear),
)


def _slack(true_opt):
    """Relative slack, floored at 1 — the yardstick the audited sites now use."""
    return REL_TOL * max(1.0, abs(true_opt))


@pytest.mark.smoke
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name,builder", MODELS, ids=[m[0] for m in MODELS])
def test_the_certificate_does_not_depend_on_the_objectives_scale(name, builder, scale):
    """Neither certificate invariant may break as the objective is scaled up."""
    model, true_opt = builder(scale)
    result = model.solve(time_limit=30.0)
    slack = _slack(true_opt)

    checked = 0

    # The dual bound is a claim that nothing in the box beats it. A bound above
    # the true optimum is a false certificate, and it is what every unsound site
    # in this class ultimately produces.
    if result.bound is not None and math.isfinite(result.bound):
        checked += 1
        assert result.bound <= true_opt + slack, (
            f"{name} at scale {scale:.0e}: dual bound {result.bound!r} exceeds the "
            f"true optimum {true_opt!r} by {result.bound - true_opt:.6e} "
            f"(slack {slack:.3e}) — a false lower bound, status={result.status}"
        )

    # An incumbent below the true optimum means an infeasible point was accepted
    # as feasible.
    if result.objective is not None and math.isfinite(result.objective):
        checked += 1
        assert result.objective >= true_opt - slack, (
            f"{name} at scale {scale:.0e}: incumbent {result.objective!r} is below "
            f"the true optimum {true_opt!r} by {true_opt - result.objective:.6e} "
            f"(slack {slack:.3e}) — an infeasible point reported feasible, "
            f"status={result.status}"
        )

    # A claimed optimum must actually be the optimum, and must bracket itself.
    if result.status == "optimal":
        checked += 2
        assert result.objective == pytest.approx(true_opt, rel=1e-4, abs=slack), (
            f"{name} at scale {scale:.0e}: status=optimal but objective "
            f"{result.objective!r} != true optimum {true_opt!r}"
        )
        assert result.bound is not None and result.bound <= result.objective + slack, (
            f"{name} at scale {scale:.0e}: status=optimal with bound "
            f"{result.bound!r} above incumbent {result.objective!r}"
        )

    # §6: a probe that checked nothing must fail rather than report a pass. A
    # solve that returns neither a bound nor an incumbent has told us nothing
    # about the certificate, and silently counting that as green is exactly how
    # this class of defect survived.
    assert checked > 0, (
        f"{name} at scale {scale:.0e}: the solve produced neither a finite bound "
        f"nor an incumbent (status={result.status}); nothing was verified"
    )


@pytest.mark.smoke
def test_the_sweep_actually_exercises_the_scale_axis():
    """Guard the probe itself: the sweep must span the range it claims to.

    A sweep silently collapsed to a single scale (an edited ``SCALES``, a
    parametrize that stopped expanding) would keep passing while testing nothing
    the ``s = 1`` control does not already cover. The audited defects are
    invisible below ``‖·‖ ≈ 1e9``, so a probe that does not reach past it is not
    this probe.
    """
    assert min(SCALES) == 1.0, "the unscaled control must be in the sweep"
    assert max(SCALES) >= 1e12, f"the sweep tops out at {max(SCALES):.0e}, too low to bite"
    assert len(set(SCALES)) == len(SCALES) >= 4, "the sweep must span several decades"
    assert len(MODELS) >= 4, "one relaxation route is not a class-level probe"
    # The slack must track the scale; an absolute slack is the audited defect.
    assert _slack(-1e12) > _slack(-1.0) * 1e11, "the probe's own tolerance is not scale-aware"
    assert _slack(-1.0) == pytest.approx(REL_TOL), "the floor must leave an O(1) problem alone"
