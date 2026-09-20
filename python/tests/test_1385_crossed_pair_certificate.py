"""#1385 — a bound past its own incumbent is not a certificate.

The hole this closes was in #1383's own guard, shipped hours earlier.
``_withhold_stale_certificate`` re-tests the final ``(objective, bound)`` pair
with ``_recertify_gap_closed``, which is ``_gap_values_converged``, which
computes::

    abs_gap = max(0.0, ub - lb)

so an **inverted** pair clamps to gap 0 and reads as *converged*. That is why
``_recertify_gap_closed``'s docstring says "The caller still owns the
on-correct-side guard" — and the caller did not own it. The guard was therefore
blind to the single case it most needed to catch.

Found by an adversarial cross-configuration round on the vendored corpus: the
same instance solved under six configurations, requiring every configuration's
dual bound to sit on the correct side of every other configuration's attained
incumbent. On ``nvs14`` (MINIMISE) with ``nlp_bb=True`` at a 60 s limit::

    status="optimal"   objective=-40358.154769   bound=+314.237382
    bound_valid=True   gap_certified=True        bound_source="bnb_tree"

A *lower* bound 40672 **above** the incumbent it is reported against — the
literal CLAUDE.md §1 invariant — published as a certificate. Two independent
routes (the default path and NLP-BB itself) attain -40358.154769, so the
incumbent is real and the bound is not.

**Scope, stated honestly.** The crossing is produced further up: the NLP-BB root
bound on this model is wrong, not merely loose (``root_bound`` is the same
+314.237382). This file pins only what belongs at the assembly boundary —
refusing to certify, and refusing to publish, a pair that contradicts itself,
*whatever produced it*. The wrong root bound is not fixed here and is not
covered by these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _guard():
    """Imported lazily so this module still COLLECTS on a tree without the fix."""
    from discopt.solver import _withhold_stale_certificate

    return _withhold_stale_certificate


# ── the guard, on the pair that fooled it ──────────────────────────────────


def test_a_crossed_minimise_pair_is_not_certified():
    """The exact numbers from nvs14, fed to the guard on their own."""
    status, gap, certified, bound = _guard()(
        "optimal",
        -40358.154769231216,  # attained by two independent routes
        314.23738194263956,  # a LOWER bound above it
        0.0,  # the gap the clamp produced
        True,
        False,  # minimise
        1e-4,
        1e-6,
        "unit",
    )
    assert certified is False, "a lower bound above its own incumbent was certified"
    assert status == "feasible", f"status stayed {status!r}"
    assert bound is None, f"the invalid bound {bound} was published anyway"
    assert gap is None, f"a gap {gap} was published with no bound"


def test_a_crossed_maximise_pair_is_not_certified():
    """A MAXIMISE bound is an UPPER bound; below the incumbent is the crossing."""
    status, gap, certified, bound = _guard()(
        "optimal", 100.0, 40.0, 0.0, True, True, 1e-4, 1e-6, "unit"
    )
    assert certified is False and status == "feasible"
    assert bound is None and gap is None


def test_the_clamp_is_why_this_was_needed():
    """Pin the mechanism, so a future refactor cannot quietly restore it.

    ``_gap_values_converged`` reports a crossed pair as converged. If that ever
    stops being true this test fails loudly and the guard above can be revisited
    — rather than the guard silently becoming dead code.
    """
    from discopt.solver import _gap_values_converged

    # ub=-40358 (incumbent), lb=+314 (bound): inverted by 40672.
    assert _gap_values_converged(-40358.154769231216, 314.23738194263956, 1e-4, 1e-6), (
        "the clamp no longer reports a crossed pair as converged"
    )


def test_a_pair_on_the_correct_side_is_untouched():
    """The fix must not cost a single legitimate certificate."""
    g = _guard()
    # exactly converged
    assert g("optimal", 5.0, 5.0, 0.0, True, False, 1e-4, 1e-6, "unit") == (
        "optimal",
        0.0,
        True,
        5.0,
    )
    # converged on the relative arm, bound below the incumbent as it must be
    status, gap, certified, bound = g(
        "optimal", 1.0, 1.0 - 5e-5, 0.0, True, False, 1e-4, 1e-6, "unit"
    )
    assert certified is True and status == "optimal" and bound == pytest.approx(1.0 - 5e-5)


def test_bound_inversion_noise_is_not_treated_as_a_crossing():
    """A few ULPs of inversion is rounding, not a broken certificate.

    The guard uses the repo's shared ``bound_inversion_tolerance``, so it agrees
    with OA, AMP and the LOA path on where noise ends.
    """
    status, gap, certified, bound = _guard()(
        "optimal", 1000.0, 1000.0 + 1e-9, 0.0, True, False, 1e-4, 1e-6, "unit"
    )
    assert certified is True, "a few ULPs of inversion were treated as a crossing"
    assert status == "optimal" and bound is not None


# ── end to end, on the instance that found it ──────────────────────────────


@pytest.mark.slow
def test_nvs14_nlp_bb_does_not_publish_a_crossed_bound():
    """The whole point: no route may publish a bound past its own incumbent."""
    from discopt.modeling.core import from_nl

    r = from_nl(os.path.join(NL, "nvs14.nl")).solve(time_limit=120, nlp_bb=True)

    # Asserted unconditionally, so a budget-starved run still checks the invariant
    # rather than skipping into a vacuous pass (CLAUDE.md §6): the three bound
    # fields must agree with each other whatever the run achieved.
    assert r.status != "error", f"status={r.status}"
    if r.bound is None:
        assert not getattr(r, "bound_valid", False), "bound_valid=True beside bound=None"
        assert r.gap is None, f"a gap {r.gap} beside bound=None"
    assert not (r.status == "optimal" and r.bound is None), (
        "status=optimal with no bound is not a certificate either"
    )
    if r.objective is None:
        return  # no pair exists, so nothing can cross

    assert r.objective == pytest.approx(-40358.154769, rel=1e-6), (
        f"the incumbent moved: {r.objective}"
    )
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (
            f"published a lower bound {r.bound} above the incumbent {r.objective}"
        )
