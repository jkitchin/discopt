"""Robustness of the pure-Rust simplex equilibration scaling.

Guards the `MAX_LINE_RANGE` noise floor in the equilibration (crate
`lp::simplex::scaling`): it must (a) keep a ~1e-16 noise coefficient from
over-scaling a column — the bug that returned a `[0,1]` variable at -1 — and
(b) NOT corrupt LPs with genuinely wide (but real) coefficient ranges, which the
floor could in principle drop. Validated against HiGHS as the oracle.
"""

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_simplex import solve_lp as simplex_lp

# This robustness gate validated the simplex on *ill-conditioned* LPs against an
# independent exact oracle. HiGHS was that oracle and has been removed (issue
# #356); the POUNCE IPM is unsuitable here — its analytic-center objective can
# drift on exactly these wide-coefficient LPs (#145), so it cannot serve as the
# exact reference. Skip until an independent exact oracle is wired back in.
pytest.skip(
    "exact external LP oracle (HiGHS) removed in issue #356; the ill-conditioned "
    "simplex-scaling cross-check needs an independent exact oracle",
    allow_module_level=True,
)

highs_lp = None  # unreachable; retained so static references below resolve.


def _agree(rs, rh, tol=1e-4):
    """Simplex either matches the HiGHS optimum or soundly declines (non-optimal).

    A sound decline (the simplex returns a non-OPTIMAL status rather than a wrong
    `OPTIMAL`) is acceptable — callers recover. Only a wrong certified optimum or
    a bound-violating solution fails.
    """
    if rh.status != SolveStatus.OPTIMAL:
        return True
    if rs.status != SolveStatus.OPTIMAL:
        return True  # sound decline -> recovery path
    return abs(rs.objective - rh.objective) <= tol * (1.0 + abs(rh.objective))


def test_noise_entry_column_solves_correctly():
    """A column carrying a ~1e-16 noise coefficient must not be over-scaled.

    Without the dynamic-range floor the column factor blows up ~1e8, pinning the
    variable's scaled bounds to ~0; the scaled point unscales out of [0, 1].
    """
    # min -x0 - x1  s.t.  x0 + x1 <= 1 ; a noise row barely touches x0.
    c = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0], [1e-16, 0.0]])
    b = np.array([1.0, 1.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    rs = simplex_lp(c, A_ub=A, b_ub=b, bounds=bounds)
    rh = highs_lp(c, A_ub=A, b_ub=b, bounds=bounds)
    assert _agree(rs, rh)
    if rs.status == SolveStatus.OPTIMAL:
        x = np.asarray(rs.x)
        assert np.all(x >= -1e-6) and np.all(x <= 1 + 1e-6), f"out-of-box: {x}"


@pytest.mark.parametrize("seed", range(15))
def test_genuine_wide_range_matches_highs(seed):
    """Real coefficients spanning ~15 orders of magnitude: never a wrong optimum.

    The noise floor could in principle ignore a *genuine* tiny coefficient; this
    ensures that never yields a wrong certified objective or an off-bound point.
    """
    rng = np.random.default_rng(seed)
    n = 5
    A = rng.uniform(-1, 1, (4, n)) * (10.0 ** rng.integers(-11, 4, (4, n)))
    b = rng.uniform(5, 20, 4)
    c = rng.uniform(-1, 1, n)
    bounds = [(0.0, 10.0)] * n
    rs = simplex_lp(c, A_ub=A, b_ub=b, bounds=bounds)
    rh = highs_lp(c, A_ub=A, b_ub=b, bounds=bounds)
    assert _agree(rs, rh), f"simplex {rs.objective} vs highs {rh.objective}"
    if rs.status == SolveStatus.OPTIMAL:
        x = np.asarray(rs.x)
        assert np.all(x >= -1e-6) and np.all(x <= 10 + 1e-6), f"out-of-box: {x}"
