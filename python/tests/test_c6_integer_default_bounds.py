"""Regression test for correctness issue C-6.

The modeling API used to give ``m.integer(name)`` a silent default bound box of
``[0, 1e6]``. A model whose optimum needs a negative or ``> 1e6`` integer was
silently truncated and the truncated-box optimum reported as certified optimal —
i.e. the solver silently solved a *different* problem than the user posed
(CLAUDE.md §3: no silent approximations).

The fix keeps a finite default (so downstream B&B still gets a bounded integer),
but applies it **loudly** — a ``UserWarning`` is emitted whenever discopt has to
supply a default integer bound because the user did not specify one. Critically,
a user-provided ``lb``/``ub`` is honored **exactly** and never triggers the
warning, and the default must never override or narrow a user-provided bound.

These tests call the modeling API directly (sub-second, no ``Model.solve()``).
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model


@pytest.mark.smoke
def test_user_negative_integer_bounds_honored_exactly():
    """A user-declared negative integer lb/ub is stored verbatim (fix-before/after)."""
    m = Model()
    n = m.integer("n", lb=-5, ub=10)
    assert float(n.lb) == -5.0
    assert float(n.ub) == 10.0


@pytest.mark.smoke
def test_user_large_integer_ub_not_clamped():
    """A user ub above the old 1e6 default is honored, not silently clamped down."""
    m = Model()
    n = m.integer("big", lb=0, ub=5_000_000)
    assert float(n.ub) == 5_000_000.0


@pytest.mark.smoke
def test_user_provided_bounds_do_not_warn():
    """Fully specified integer bounds must not emit the default-bound warning."""
    m = Model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning becomes a failure
        n = m.integer("n", lb=-3, ub=3)
    assert float(n.lb) == -3.0
    assert float(n.ub) == 3.0


@pytest.mark.smoke
def test_unspecified_integer_bounds_warn_loudly():
    """An unspecified integer bound gets the finite default but WARNS loudly.

    This is the core C-6 guard: silently imposing ``[0, 1e6]`` is the bug. The
    default may still be applied (B&B needs a bounded integer) but it must not
    be silent.
    """
    m = Model()
    with pytest.warns(UserWarning, match="default"):
        n = m.integer("n")
    # Default still finite so downstream B&B has a bounded integer.
    assert np.isfinite(float(n.lb))
    assert np.isfinite(float(n.ub))


@pytest.mark.smoke
def test_partial_bounds_default_only_missing_side():
    """Specifying only lb warns about the defaulted ub, honors lb exactly."""
    m = Model()
    with pytest.warns(UserWarning, match="default"):
        n = m.integer("n", lb=-2)
    assert float(n.lb) == -2.0  # user side honored exactly, not overridden
    assert np.isfinite(float(n.ub))


@pytest.mark.smoke
def test_indexed_integer_negative_bounds_honored():
    """The indexed (``over=``) integer path also honors negative user bounds."""
    m = Model()
    s = dm.Set("s", [1, 2, 3])
    n = m.integer("n", over=s, lb=-4, ub=7)
    flat = n.flat
    assert float(np.min(flat.lb)) == -4.0
    assert float(np.max(flat.ub)) == 7.0
