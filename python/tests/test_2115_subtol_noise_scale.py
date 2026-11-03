"""#2115 follow-up: rounding residue of ``y`` must not pass as a real sub-``tol``
reduced cost.

#1595's re-pricing (``subtol_scan``) treats a wrong-signed reduced cost below the
absolute pricing tolerance as real when it exceeds ``1e-6`` of the terms it was
computed from, ``|c_j| + sum_i |a_ij y_i|``. A single-entry slack has
``d_j = -a y_i``: one term, so it is always "relative size 1", however small
``y_i`` is. On the MINLPLib ``m6`` final OA master
(``data/oa_masters/m6_final_master_2115.npz``) the duals carry ``y_i`` of
1e-15 residue against ``||y||_inf`` of 12..192 on rows whose slacks are open. Those
columns' room is infinite, so the re-pricing entered them; the unblocked ray had
``c^T d = 0`` exactly and was rightly refused, and the LP came back ``Numerical``
instead of the ``Optimal`` its dual had already reached. Measured: 13 such
verdicts, every one of the 13 sub-``tol`` pivots noise-driven.

``DISCOPT_SUBTOL_NOISE_GLOBAL`` (default ON; graduated by owner decision as a
verdict fix, performance-plan 104.1) also measures the reduced cost against
``||a_j||_1 ||y||_inf``, the scale of ``y``'s error as it reaches ``d_j``. A column
below it becomes a *suspect* (the #1953 path: settled by the NS bound and one step
of dual refinement), so a genuine sub-``tol`` reduced cost still enters. The Rust
unit test ``subtol_noise_is_measured_against_y_scale_2115`` pins the scan on a 2x2
LP, including the #1595 witness shape staying a candidate; this pins the master.

Each arm runs in its own process: the flag is a Rust ``OnceLock``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")
MASTER = os.path.join(DATA, "oa_masters", "m6_final_master_2115.npz")
M6_OPTIMUM = 82.2568769100

_RUNNER = r"""
import json, sys
import numpy as np, scipy.sparse as sp
from discopt.solvers import milp_simplex
from discopt._rust import profile_counters_py, profile_reset_py

d = np.load(sys.argv[1])
A = sp.csr_matrix((d["A_ub_data"], d["A_ub_indices"], d["A_ub_indptr"]),
                  shape=tuple(d["A_ub_shape"]))
profile_reset_py()
r = milp_simplex.solve_milp(
    c=d["c"], A_ub=A, b_ub=d["b_ub"], bounds=list(zip(d["lb"], d["ub"])),
    integrality=d["integrality"], time_limit=120.0,
    gap_tolerance=float(d["gap_tolerance"]), mip_start=d["mip_start"],
)
c = dict(profile_counters_py())
print("RESULT " + json.dumps({
    "status": r.status.value, "bound": r.bound,
    "numerical": int(c.get("LpVerdictNumerical", 0)),
    "subtol_pivots": int(c.get("SubtolPivots", 0)),
    "ray_obj_reject": int(c.get("UnboundedRejectObjective", 0)),
    "repairs": int(c.get("WarmSubtolRepair", 0)),
}), flush=True)
"""


def _solve(flag: str) -> dict:
    env = {**os.environ, "DISCOPT_PROFILE": "1", "DISCOPT_SUBTOL_NOISE_GLOBAL": flag}
    env.pop("DISCOPT_NS_PROJECT", None)  # the shipped default (ON)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, MASTER],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[len("RESULT ") :])
    raise AssertionError(f"no result (rc={proc.returncode}): {proc.stderr[-800:]}")


@pytest.mark.slow
@pytest.mark.correctness
def test_m6_master_noise_reduced_costs_are_not_pivoted_on():
    """Premise (OFF): noise-driven sub-``tol`` pivots end in refused rays and
    ``Numerical`` LPs. Fix (ON): none of either, the re-pricing still ran (anti-
    vacuity, §6), and the master certifies with the bound never above the optimum.

    Measured (dev machine, unloaded): OFF 13 ``Numerical`` / 13 sub-tol pivots /
    8 refused rays; ON 0 / 0 / 0; both ``optimal`` in ~9 s (25 406 vs 25 392 nodes).
    """
    off = _solve("0")
    assert off["subtol_pivots"] > 0 and off["numerical"] > 0, off
    assert off["ray_obj_reject"] > 0, off

    on = _solve("1")
    assert on["repairs"] > 0, on  # the sub-tol re-pricing ran
    assert on["numerical"] == 0, on
    assert on["subtol_pivots"] == 0, on
    assert on["ray_obj_reject"] == 0, on
    assert on["status"] == "optimal", on
    assert on["bound"] <= M6_OPTIMUM + 1e-6 * (1.0 + M6_OPTIMUM), on
