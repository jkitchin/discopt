"""#2142 B: GDPlib ``cstr`` big-M node LPs the warm dual could not decide.

``data/cstr_bigm_warm_node_lps.npz`` holds 16 node LPs captured from the native
spatial kernel on ``cstr`` (``gdp_method="big-m"``), each with the basis the kernel
warm-started it from: the first 12 (by node id) of the 353 the kernel left
``numerical`` in a 120 s solve, and 4 it solved ``optimal`` (controls). Every one
has a HiGHS reference objective (presolve off, ``small_matrix_value=1e-12`` so no
entry is dropped).

They are replayed through ``solve_lp_warm_csc`` -- the kernel's own warm entry,
with the kernel's ``warm_flip_boxed`` -- in a fresh interpreter per arm, because
``DISCOPT_LP_DUAL_REBUILD`` is read once per process.

The B0 counters named the exit: the dual ratio test comes back empty with the
leaving row off its bound by ~1e-8 (against the loop's ``tol`` = 1e-9), a fresh
factorization reproduces that residual, and the Farkas ray does not certify, so
the LP is handed to the cold primal, which ends ``Numerical``. With the flag on
the dual rebuilds, tolerates the immovable row inside ``1e-7*(1+|bound|)`` and
continues; an ``Optimal`` reached that way must pass the original-LP audit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "data" / "cstr_bigm_warm_node_lps.npz"

_WORKER = r"""
import json, sys
import numpy as np
import discopt._rust as R
d = np.load(sys.argv[1])
out = []
i = 0
while f"lp{i}_m" in d.files:
    g = lambda k: d[f"lp{i}_{k}"]
    m, n = int(g("m")), int(g("n"))
    st, x, obj, _it, _cs, _bv, dual, _ray = R.solve_lp_warm_csc_py(
        np.asarray(g("c"), float), m, n, np.asarray(g("col_ptr"), np.int64),
        np.asarray(g("row_idx"), np.int64), np.asarray(g("vals"), float),
        np.asarray(g("b"), float), np.asarray(g("l"), float), np.asarray(g("u"), float),
        np.asarray(g("start_col_status"), np.int8), np.asarray(g("start_basic_vars"), np.int64),
        warm_flip_boxed=True,
    )
    out.append({"status": st, "obj": float(obj), "x": [float(v) for v in x],
                "n_dual": len(dual)})
    i += 1
print(json.dumps(out))
"""


def _replay(flag: str) -> list[dict]:
    env = {**os.environ, "DISCOPT_LP_DUAL_REBUILD": flag}
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, str(FIXTURE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _audit_ok(d, i: int, x: np.ndarray) -> bool:
    """The original-LP audit: rows and bounds within ``1e-6 * (1 + |rhs or bound|)``."""
    import scipy.sparse as sp

    m, n = int(d[f"lp{i}_m"]), int(d[f"lp{i}_n"])
    A = sp.csc_matrix((d[f"lp{i}_vals"], d[f"lp{i}_row_idx"], d[f"lp{i}_col_ptr"]), shape=(m, n))
    b, lo, hi = d[f"lp{i}_b"], d[f"lp{i}_l"], d[f"lp{i}_u"]
    rows = np.all(np.abs(A @ x - b) <= 1e-6 * (1.0 + np.abs(b)))
    inf = 1e20
    lo_ok = np.all((lo <= -inf) | (x >= lo - 1e-6 * (1.0 + np.abs(lo))))
    hi_ok = np.all((hi >= inf) | (x <= hi + 1e-6 * (1.0 + np.abs(hi))))
    return bool(rows and lo_ok and hi_ok)


@pytest.fixture(scope="module")
def arms():
    return _replay("0"), _replay("1")


def test_dual_rebuild_decides_cstr_bigm_node_lps(arms):
    d = np.load(FIXTURE)
    off, on = arms
    kinds = [str(d[f"lp{i}_kind"]) for i in range(len(off))]
    ref = [float(d[f"lp{i}_highs_obj"]) for i in range(len(off))]
    num = [i for i, k in enumerate(kinds) if k == "numerical"]
    ctrl = [i for i, k in enumerate(kinds) if k == "control"]
    executed = 0
    assert len(num) == 12 and len(ctrl) == 4
    executed += 1

    # Before: the legacy path leaves the class undecided (pins the defect).
    off_decided = sum(off[i]["status"] in ("optimal", "infeasible") for i in num)
    assert off_decided <= 2, f"flag OFF decided {off_decided}/12; the fixture no longer pins #2142"
    executed += 1

    # After: most are decided, and every decision is HiGHS's optimum on a point the
    # original-LP audit accepts. None may be called infeasible: HiGHS solves all 12.
    on_opt = [i for i in num if on[i]["status"] == "optimal"]
    assert len(on_opt) >= 10, f"flag ON decided {len(on_opt)}/12"
    executed += 1
    for i in num:
        assert on[i]["status"] != "infeasible", f"lp{i}: false infeasible"
        executed += 1
    for i in on_opt:
        assert abs(on[i]["obj"] - ref[i]) <= 1e-6 * (1.0 + abs(ref[i])), (i, on[i]["obj"], ref[i])
        assert _audit_ok(d, i, np.asarray(on[i]["x"])), f"lp{i}: point fails the audit"
        assert on[i]["n_dual"] == int(d[f"lp{i}_m"]), "row duals exported for the NS bound"
        executed += 3

    # Controls never reach the rebuild: identical in both arms, and HiGHS's optimum.
    for i in ctrl:
        assert off[i]["status"] == on[i]["status"] == "optimal"
        assert off[i]["obj"] == on[i]["obj"] and off[i]["x"] == on[i]["x"]
        assert abs(on[i]["obj"] - ref[i]) <= 1e-6 * (1.0 + abs(ref[i]))
        executed += 3

    print(
        f"#2142 node-LP replay: OFF decided {off_decided}/12, ON optimal {len(on_opt)}/12; "
        f"executed assertions: {executed}"
    )
    assert executed >= 1 + 1 + 1 + 12 + 3 * 10 + 3 * 4
