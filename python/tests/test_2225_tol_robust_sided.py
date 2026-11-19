"""#2225: an IIS probe whose node LP is only tolerance-feasible at the symmetric box's
edge is decided, not left as an ``unresolved_floor``.

``data/iis_2225_bell3ainf_sided_probe.npz`` is one MIPcc27 ``bell3ainf`` IIS probe (the
model and the probe's rows, bounds and integrality; HiGHS proves it infeasible). The
HiGHS search reached a node with every integer column fixed whose rows are infeasible
exactly and infeasible over the half-tolerance candidate box, but feasible in the full
tolerance box only at its edge: row slacks ``3e-6`` above ``u = 0``, where the
acceptance test (``point_verified``) allows ``1e-6`` row plus ``1e-6`` bound. The full
box charged each slack the bound tolerance of its larger side (``max(|l|, |u|) = 1``) on
both sides. So the node's point was never acceptable and its emptiness never provable;
the search kept it as an ``unresolved_floor`` and the solve came back ``node_limit``
with ``stop_reason == "none"``: no budget spent, nothing decided. In the IIS engine such
a probe keeps its element (67 of 371 probes of one run; IIS 174 vs HiGHS's 130).

``DISCOPT_MILP_TOL_ROBUST_SIDED`` (default ON) widens each slack by its own side's
tolerance, the exact projection of the accepted set, and proves an infeasible LP over
that box. The env var is read once per process, so each arm runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

DATA = Path(__file__).parent / "data" / "iis_2225_bell3ainf_sided_probe.npz"

_PROBE = r"""
import json, sys, numpy as np, scipy.sparse as sp
from discopt.solvers.milp_handle import MilpHandle
z = np.load(sys.argv[1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
h = MilpHandle(np.zeros(A.shape[1]), A, z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"],
               z["is_int"])
r = h.solve(time_limit=60.0, presolve=True)
print(json.dumps({"status": r.status, "stop_reason": r.stop_reason, "nodes": r.nodes}))
"""


def _solve(sided: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_MILP_TOL_ROBUST_SIDED"}
    if sided is not None:
        env["DISCOPT_MILP_TOL_ROBUST_SIDED"] = sided
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(DATA)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_sided_box_decides_the_probe_infeasible():
    """Default (sided ON): the probe is proved infeasible, as HiGHS proves it. Before the
    fix this solve returned ``node_limit`` with no budget spent (``stop_reason`` none)."""
    r = _solve(None)
    assert r["status"] == "infeasible", r


def test_opt_out_reproduces_the_undecided_probe():
    """``=0`` restores the symmetric box: the pre-fix outcome, a ``node_limit`` stop that
    no budget caused. Pins that the fixture still exercises the defect, so the test above
    is not passing on a probe that stopped needing the sided box."""
    r = _solve("0")
    assert r["status"] == "node_limit", r
    assert r["stop_reason"] == "none", r


@pytest.mark.requires_highs
def test_highs_agrees_the_probe_is_infeasible():
    """The oracle: HiGHS proves the same probe infeasible."""
    pytest.importorskip("highspy")
    from discopt._iis_backend import INFEASIBLE, make_backend

    z = np.load(DATA)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    b = make_backend("highs", A, z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"], z["is_int"])
    b.configure(z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"], z["is_int"])
    import time

    assert b.run("off", time.perf_counter() + 60.0) == INFEASIBLE


def test_handle_result_carries_the_stop_reason():
    """``MilpHandle.solve`` records the driver's counters like every other binding, so
    ``HandleResult.stop_reason`` names the budget: ``"none"`` for a decided solve and
    ``"nodes"`` for one the node cap stopped. Before the fix the handle recorded nothing
    (``milp_last_solve_stats()`` stayed ``None``), so an IIS probe's ``node_limit`` could
    not be told apart from a give-up."""
    from discopt._rust import milp_last_solve_stats
    from discopt.solvers.milp_handle import MilpHandle

    z = np.load(DATA)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    # The probe itself, decided (infeasible) under the default sided box.
    probe = MilpHandle(
        np.zeros(A.shape[1]),
        A,
        z["row_lo"],
        z["row_hi"],
        z["col_lb"],
        z["col_ub"],
        z["is_int"],
    )
    r = probe.solve(time_limit=60.0, presolve=True)
    # The wrapper read (and cleared) the stats this solve recorded.
    assert milp_last_solve_stats() is None
    assert r.stop_reason == "none" and r.status == "infeasible", r

    # A small knapsack-like MILP that needs more than one node: capped at 1 node.
    rng = np.random.default_rng(7)
    n = 30
    w = rng.integers(5, 40, n).astype(float)
    h = MilpHandle(
        -rng.integers(1, 30, n).astype(float),
        sp.csc_matrix(w.reshape(1, -1)),
        [-np.inf],
        [0.5 * w.sum() + 0.5],
        np.zeros(n),
        np.ones(n),
        np.ones(n, dtype=bool),
    )
    capped = h.solve(max_nodes=1, presolve=False, heuristics=False)
    assert capped.status == "node_limit", capped
    assert capped.stop_reason == "nodes", capped
