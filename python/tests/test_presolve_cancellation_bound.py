"""MILP presolve must not certify a bound above a feasible point (cancellation drops).

The captured portfol_roundlot OA master (``data/oa_masters/portfol_roundlot_master13.npz``,
see the README there) links continuous weights to integer lots, ``n_j = a_j x_j`` with
``a_j`` up to 1e5, under OA cuts with coefficients ~1. Presolve aggregates the ``x_j``
out, and the cancellation in the substituted cut rows left coefficients of ~1e-12 on
integers up to 1e5. Removing those as zero, without relaxing the row, cut off a point
feasible for the original master: the default solve certified a dual bound of
0.0282902567 against ``z_star`` at 0.0282902109 (4.6e-8 above it), and the OA run
built on that master never closed (300 s against 4.3 s before presolve was on).

``z_star`` is the witness: its integers ``[87462, 0, 0, 0, 0, 8478, 1793, 624]`` with
the continuous part solved exactly, re-verified below before it is used (CLAUDE.md
§6). The slack is absolute and measured: the solve's bound now lands 2.3e-10 below
``z_star``'s objective, against a defect of 4.6e-8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers.milp_simplex import solve_milp

MASTER = Path(__file__).parent / "data" / "oa_masters" / "portfol_roundlot_master13.npz"

# Absolute, in objective units: 200x below the 4.6e-8 defect, 1e-8/2.3e-10 = 43x above
# the measured legitimate margin.
SLACK = 1e-8


def _load():
    d = np.load(MASTER)
    kw = dict(
        c=d["c"],
        bounds=list(zip(d["lb"].tolist(), d["ub"].tolist())),
        integrality=d["integrality"],
        gap_tolerance=float(d["gap_tolerance"]),
    )
    for mat, rhs in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        kw[mat] = sp.csr_matrix(
            (d[f"{mat}_data"], d[f"{mat}_indices"], d[f"{mat}_indptr"]),
            shape=tuple(d[f"{mat}_shape"]),
        )
        kw[rhs] = d[rhs]
    return kw, d["z_star"], float(d["z_star_objective"])


@pytest.mark.smoke
def test_the_presolved_master_is_not_certified_above_a_feasible_point():
    problem, z, attained = _load()

    # The witness must really be feasible, at tolerances far below the defect.
    assert np.max(problem["A_ub"] @ z - problem["b_ub"]) <= 1e-12
    assert np.max(np.abs(problem["A_eq"] @ z - problem["b_eq"])) <= 1e-9
    lo = np.array([b[0] for b in problem["bounds"]])
    hi = np.array([b[1] for b in problem["bounds"]])
    assert np.all(z >= lo) and np.all(z <= hi)
    ints = np.nonzero(problem["integrality"])[0]
    assert np.all(z[ints] == np.round(z[ints]))
    assert float(problem["c"] @ z) == pytest.approx(attained, abs=1e-15)

    res = solve_milp(time_limit=60.0, **problem)

    assert res.bound is not None
    assert res.bound <= attained + SLACK, (
        f"dual bound {res.bound!r} above a point of {attained!r} feasible for this "
        "very master -- a false certificate"
    )
    if res.status.value == "optimal":
        assert res.objective is not None and res.objective <= attained + 3e-6, (
            "certified optimal above the witness by more than the requested gap"
        )
