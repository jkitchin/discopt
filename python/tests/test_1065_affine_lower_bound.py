"""#1065 — ``_affine_lower_bound`` must not discard a bound on ``0 * inf``.

``np.where(vec >= 0, vec * lo_box, vec * hi_box)`` evaluates BOTH branches, so a
variable that does not appear in the affine form (coefficient 0) but is declared
unbounded contributed ``0 * inf = NaN``. The ``isfinite`` guard then rejected the
whole bound and numpy emitted a RuntimeWarning into user output. Not a soundness
bug (it returned ``None`` = "unknown"), but it fired on nearly every affine form
in the big-M/GDP families, which declare ``lb=0, ub=inf`` on most continuous
variables (#1061).

Each test pins one direction, including the anti-vacuity control that a
genuinely unbounded *contributing* variable still yields ``None`` (§6) — without
it, "returns a finite bound" could be satisfied by a function that ignores
infinities altogether.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.patterns import _affine_lower_bound


def test_zero_coefficient_on_unbounded_lower_endpoint_keeps_the_bound():
    """The bound-discarding case: a bystander declared ``lb=-inf``.

    A zero coefficient satisfies ``vec >= 0``, so the *selected* branch is
    ``vec * lo_box`` — the NaN lands in ``contrib`` itself and the finiteness
    guard throws the bound away. (With the more common ``lb=0, ub=inf``
    bystander the NaN is formed in the discarded ``vec * hi_box`` branch, so the
    damage is only the RuntimeWarning; the next test pins that.)
    """
    m = dm.Model("zero_coeff_unbounded_below")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.continuous("z", lb=-float("inf"), ub=0.0)  # unbounded below, coefficient 0
    m.minimize(x)

    assert _affine_lower_bound(0.001 + 0.999 * x, m) == pytest.approx(0.001)


def test_zero_coefficient_on_unbounded_upper_endpoint_keeps_the_bound():
    """The corpus-common shape: ``lb=0, ub=inf`` bystanders (syn/rsyn/squfl)."""
    m = dm.Model("zero_coeff_unbounded")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.continuous("z", lb=0.0, ub=float("inf"))  # unbounded above, coefficient 0
    m.minimize(x)

    assert _affine_lower_bound(0.001 + 0.999 * x, m) == pytest.approx(0.001)


def test_no_runtime_warning_from_the_zero_times_inf_product():
    m = dm.Model("no_warning")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.continuous("z", lb=0.0, ub=float("inf"))
    m.minimize(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _affine_lower_bound(0.001 + 0.999 * x, m) == pytest.approx(0.001)


def test_unbounded_contributing_variable_still_returns_none():
    """Anti-vacuity control: the guard the fix must NOT weaken.

    ``x`` has a positive coefficient and no lower bound, so the affine form is
    genuinely unbounded below and ``None`` ("unknown") is the only sound answer.
    """
    m = dm.Model("unbounded_contributor")
    x = m.continuous("x", lb=-float("inf"), ub=1.0)
    m.minimize(x)

    assert _affine_lower_bound(1.0 + x, m) is None


def test_negative_coefficient_uses_the_upper_endpoint():
    """The ``c < 0`` branch still minimizes at ``ub`` (endpoint selection intact)."""
    m = dm.Model("negative_coeff")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.continuous("z", lb=0.0, ub=float("inf"))
    m.minimize(x)

    # min of 5 - 2x over x in [0, 2] is at x = 2 -> 1.0
    assert _affine_lower_bound(5.0 - 2.0 * x, m) == pytest.approx(1.0)


def test_negative_coefficient_on_an_unbounded_upper_endpoint_returns_none():
    """Second anti-vacuity control, for the ``hi_box`` side of the selection."""
    m = dm.Model("negative_coeff_unbounded")
    x = m.continuous("x", lb=0.0, ub=float("inf"))
    m.minimize(x)

    assert _affine_lower_bound(5.0 - 2.0 * x, m) is None


def test_many_unbounded_bystanders_do_not_poison_the_sum():
    """The corpus shape: a small form inside a model full of unbounded flows."""
    m = dm.Model("many_bystanders")
    x = m.continuous("x", lb=0.0, ub=1.0)
    for i in range(25):
        m.continuous(f"f{i}", lb=0.0, ub=float("inf"))
    m.minimize(x)

    bound = _affine_lower_bound(2.0 + 3.0 * x, m)
    assert bound is not None
    assert bound == pytest.approx(2.0)
    assert np.isfinite(bound)
