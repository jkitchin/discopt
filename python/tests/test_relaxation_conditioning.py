"""Unit tests for numerical-conditioning sanitization of the MILP relaxation.

``sanitize_relaxation_for_conditioning`` (issue #138) makes a catastrophically
scaled relaxation solvable by the LP backend by dropping non-finite / huge
constraint rows and clamping huge variable bounds to +/-inf. Both transforms only
*relax* the feasible set, so the LP value stays a valid lower bound for a
minimization — never above the true optimum. These tests lock that contract
without the slow ``.nl`` solves.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import scipy.sparse as sp
from discopt._jax.milp_relaxation import (
    _RELAX_NUMERIC_CAP,
    MilpRelaxationModel,
    sanitize_relaxation_for_conditioning,
)


def _model(A, b, bounds, c=None):
    n = len(bounds)
    return MilpRelaxationModel(
        c=np.zeros(n) if c is None else np.asarray(c, dtype=float),
        A_ub=sp.csr_matrix(np.asarray(A, dtype=float)) if A is not None else None,
        b_ub=np.asarray(b, dtype=float) if b is not None else None,
        bounds=bounds,
    )


def test_well_scaled_relaxation_is_unchanged():
    """No row reaches the cap -> rows and bounds pass through untouched."""
    m = _model([[1.0, 2.0], [3.0, -1.0]], [10.0, 5.0], [(0.0, 100.0), (-50.0, 50.0)])
    s = sanitize_relaxation_for_conditioning(m)
    assert s._A_ub.shape == (2, 2)
    assert np.allclose(s._A_ub.toarray(), m._A_ub.toarray())
    assert np.allclose(s._b_ub, m._b_ub)
    assert s._bounds == m._bounds


def test_catastrophic_row_coefficient_is_dropped():
    """A row with |coef| >= cap is removed; the clean row survives."""
    big = _RELAX_NUMERIC_CAP * 10
    m = _model([[1.0, 2.0], [big, 1.0]], [10.0, 3.0], [(0.0, 100.0), (0.0, 100.0)])
    s = sanitize_relaxation_for_conditioning(m)
    assert s._A_ub.shape == (1, 2)
    assert np.allclose(s._A_ub.toarray(), [[1.0, 2.0]])
    assert np.allclose(s._b_ub, [10.0])


def test_nonfinite_row_is_dropped():
    """A row with an inf coefficient or RHS is removed."""
    m = _model([[1.0, 1.0], [np.inf, 1.0]], [10.0, 2.0], [(0.0, 10.0), (0.0, 10.0)])
    s = sanitize_relaxation_for_conditioning(m)
    assert s._A_ub.shape == (1, 2)
    # Also drop a finite-coefficient row whose RHS is non-finite.
    m2 = _model([[1.0, 1.0], [2.0, 1.0]], [10.0, np.inf], [(0.0, 10.0), (0.0, 10.0)])
    s2 = sanitize_relaxation_for_conditioning(m2)
    assert s2._A_ub.shape == (1, 2)
    assert np.allclose(s2._b_ub, [10.0])


def test_catastrophic_variable_bound_is_clamped_to_inf():
    """A finite bound of catastrophic magnitude is widened to +/-inf (a relaxation)."""
    huge = _RELAX_NUMERIC_CAP * 100
    m = _model([[1.0, 1.0]], [5.0], [(0.0, huge), (-huge, 3.0)])
    s = sanitize_relaxation_for_conditioning(m)
    assert s._bounds[0] == (0.0, np.inf)
    assert s._bounds[1] == (-np.inf, 3.0)


def test_sanitization_only_relaxes_the_bound():
    """Dropping the catastrophic row only lowers (never raises) the LP optimum, so
    the sanitized bound stays a valid lower bound."""
    big = _RELAX_NUMERIC_CAP * 10
    # minimize x0 s.t. x0 >= 4 (as -x0 <= -4) and a catastrophic (vacuous) row.
    m = _model(
        [[-1.0, 0.0], [big, big]],
        [-4.0, big],
        [(0.0, 10.0), (0.0, 10.0)],
        c=[1.0, 0.0],
    )
    s = sanitize_relaxation_for_conditioning(m)
    res = s.solve(time_limit=5, gap_tolerance=1e-6)
    assert res.status == "optimal"
    # The retained constraint x0 >= 4 gives optimum 4; dropping a row can only
    # relax, so the bound never exceeds it.
    assert res.bound is not None and res.bound <= 4.0 + 1e-6
