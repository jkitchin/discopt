"""The in-house MILP driver must not prove EMPTY a box that is feasible within its own
feasibility tolerance.

Fixture: the LP/NLP-BB OA master the convex route builds for MINLPLib
``portfol_roundlot`` (17 columns; rows ``c_i x_i - n_i = 0`` with ``c_i`` up to 1e5,
integer ``n_i`` in one-unit boxes, and ``sum x = 1``). In exact arithmetic no integer
point of its box satisfies the rows; the closest misses ``sum x = 1`` by 3.35e-7,
inside the driver's incumbent tolerance. HiGHS
returns that point (0.0282907), and so does the driver on one trajectory. On another
(the D3 lazy master with its root cut loop) the driver's exact proofs fathomed every
node and the master came back ``INFEASIBLE``: node FBBT amplified the 3.35e-7
shortfall through ``x15 = 55000 x6`` and integer rounding into an empty box, and node
LP Farkas rays were accepted at a 1e-9 margin. A box the solver would accept a point
from must not be the one it proves empty.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import milp_simplex as MS

DATA = os.path.join(
    os.path.dirname(__file__), "data", "portfol_roundlot_oa_master_tolerance_feasible.npz"
)
ROW_TOL = 1e-6  # the driver's incumbent row tolerance, relative to 1 + |b|


def _load():
    z = np.load(DATA)
    a_ub = sp.csr_matrix(
        (z["Aub_data"], z["Aub_indices"], z["Aub_indptr"]), shape=tuple(z["Aub_shape"])
    )
    a_eq = sp.csr_matrix(
        (z["Aeq_data"], z["Aeq_indices"], z["Aeq_indptr"]), shape=tuple(z["Aeq_shape"])
    )
    bounds = [(float(lo), None if np.isinf(hi) else float(hi)) for lo, hi in zip(z["lb"], z["ub"])]
    return dict(
        c=z["c"],
        A_ub=a_ub,
        b_ub=z["b_ub"],
        A_eq=a_eq,
        b_eq=z["b_eq"],
        bounds=bounds,
        integrality=z["integrality"],
    )


def test_fixture_box_is_exactly_infeasible_but_tolerance_feasible():
    """Premise: the box holds no exactly feasible integer point, and its best point
    misses only by 3.35e-7 (so the fixture keeps exercising the tolerance gap)."""
    k = _load()
    ae = k["A_eq"].toarray()
    ints = [j for j in range(len(k["c"])) if k["integrality"][j]]
    link = {}
    for r in range(1, ae.shape[0]):
        nz = np.nonzero(ae[r])[0]
        cont = [j for j in nz if not k["integrality"][j]]
        intg = [j for j in nz if k["integrality"][j]]
        assert len(cont) == 1 and len(intg) == 1 and ae[r][intg[0]] == 1.0
        link[intg[0]] = (cont[0], Fraction(ae[r][cont[0]]))
    lo = [int(k["bounds"][j][0]) for j in ints]
    hi = [int(k["bounds"][j][1]) for j in ints]
    best = None
    count = 0
    for combo in itertools.product(*[range(a, b + 1) for a, b in zip(lo, hi)]):
        total = sum(Fraction(-v) / link[j][1] for j, v in zip(ints, combo))
        miss = abs(total - 1)
        best = miss if best is None else min(best, miss)
        count += 1
    assert count == 256
    assert best != 0, "an exactly feasible point exists: the fixture lost its premise"
    assert float(best) < ROW_TOL, float(best)


@pytest.mark.parametrize("lazy", [False, True], ids=["solve_milp", "lazy_master"])
def test_driver_does_not_prove_a_tolerance_feasible_box_empty(lazy):
    k = _load()
    if lazy:
        res = MS.solve_milp_with_lazy_cuts(
            **k, time_limit=60.0, gap_tolerance=1e-4, lazy_callback=lambda x: []
        )
    else:
        res = MS.solve_milp(**k, time_limit=60.0, gap_tolerance=1e-4)
    assert res.status != MS.SolveStatus.INFEASIBLE, "a tolerance-feasible box was proved empty"
    assert res.x is not None
    x = np.asarray(res.x, dtype=np.float64)
    resid_eq = np.abs(k["A_eq"] @ x - k["b_eq"]) / (1.0 + np.abs(k["b_eq"]))
    resid_ub = (k["A_ub"] @ x - k["b_ub"]) / (1.0 + np.abs(k["b_ub"]))
    assert resid_eq.max() <= ROW_TOL and resid_ub.max() <= ROW_TOL
    # The best tolerance-feasible point HiGHS reports is 0.0282907 (minimize).
    assert res.objective <= 0.0282907 + 1e-6


def test_tol_robust_opt_out_reproduces_the_defect():
    """The fixture still exercises the defect class, and ``=0`` really turns the fix off.

    ``DISCOPT_MILP_TOL_ROBUST=0`` (read once per process, so subprocesses) restores the
    exact-only proofs. On main that made the lazy master prove this box empty
    (``INFEASIBLE``). On the highs-parity search the unfixed trajectory reaches the
    optimum before such a proof is acted on (measured 2026-11-15: ``OPTIMAL`` with every
    MILP switch tried), so the end-to-end symptom is no longer the probe. What is: on
    the default the driver still meets emptiness verdicts and infeasible node LPs that
    do not survive the acceptance tolerance -- exactly the proofs the pre-fix driver
    acted on -- and refuses or relaxes them (measured: 13 FBBT verdicts refused, 11
    relaxed re-solves), while ``=0`` leaves every such counter at zero. The mechanisms
    themselves are unit-tested in ``milp_driver.rs``
    (``robust_farkas_refuses_a_ray_inside_the_tolerance`` and the #2107 tests)."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import test_milp_tolerance_feasible_box as T\n"
        "from discopt.solvers import milp_simplex as MS\n"
        "from discopt._rust import profile_counters_py, profile_reset_py\n"
        "profile_reset_py()\n"
        "r = MS.solve_milp_with_lazy_cuts(**T._load(), time_limit=60.0, gap_tolerance=1e-4,"
        " lazy_callback=lambda x: [])\n"
        "c = dict(profile_counters_py())\n"
        "print('STATUS', r.status.name)\n"
        "print('ROBUST', c['TolRobustFbbtRefused'], c['TolRobustRelaxedSolves'])\n"
    ) % os.path.dirname(__file__)

    def run(value: str) -> tuple[str, int, int]:
        env = {**os.environ, "DISCOPT_PROFILE": "1", "DISCOPT_MILP_TOL_ROBUST": value}
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        lines = dict(ln.split(" ", 1) for ln in p.stdout.splitlines() if " " in ln)
        refused, relaxed = (int(v) for v in lines["ROBUST"].split())
        return lines["STATUS"], refused, relaxed

    status_on, refused_on, relaxed_on = run("1")
    assert status_on == "OPTIMAL"
    # The fixture still produces the non-robust proofs the fix exists for.
    assert refused_on + relaxed_on > 0, (refused_on, relaxed_on)
    status_off, refused_off, relaxed_off = run("0")
    # ...and the opt-out is consumed: none of them is refused or relaxed.
    assert (refused_off, relaxed_off) == (0, 0)
    print(f"opt-out status on this trajectory: {status_off}")
