"""#2142 B: GDPlib ``cstr`` big-M node LPs the warm dual cannot decide.

``data/cstr_bigm_warm_node_lps.npz`` holds 16 node LPs captured from the native
spatial kernel on ``cstr`` (``gdp_method="big-m"``), each with the basis the kernel
warm-started it from: the first 12 (by node id) of the 353 the kernel left
``numerical`` in a 120 s solve, and 4 it solved ``optimal`` (controls). Every one
has a HiGHS reference objective (presolve off, ``small_matrix_value=1e-12`` so no
entry is dropped).

They are replayed through ``solve_lp_warm_csc`` -- the kernel's own warm entry,
with the kernel's ``warm_flip_boxed`` -- in a fresh interpreter.

The B0 counters named the exit: the dual ratio test comes back empty with the
leaving row off its bound by ~1e-8 (against the loop's ``tol`` = 1e-9), a fresh
factorization reproduces that residual, and the Farkas ray does not certify, so
the LP is handed to the cold primal, which ends ``Numerical``.
``DISCOPT_LP_DUAL_REBUILD`` (tolerate such a row within ``1e-7*(1+|bound|)``)
decided 11 of the 12 at HiGHS's objective, and was RETIRED by its CLAUDE.md
section 5 panel: cert-clean but not net-positive (cstr big-M was the only outcome
it changed; ``launch`` ran 41% slower with 21 -> 241 ``Numerical`` LP verdicts).
See ``docs/dev/flag-retirement-audit.md`` and ``docs/dev/data/2142-flag-panel/``.
This file pins the defect, and holds the bar a future fix must meet as a strict
xfail.
"""

from __future__ import annotations

import json
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


def _replay() -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, str(FIXTURE)],
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
def replay():
    return _replay()


def _kinds(d, n):
    kinds = [str(d[f"lp{i}_kind"]) for i in range(n)]
    num = [i for i, k in enumerate(kinds) if k == "numerical"]
    ctrl = [i for i, k in enumerate(kinds) if k == "control"]
    return num, ctrl


def test_cstr_bigm_node_lps_pin_legacy_verdicts(replay):
    """The defect, pinned: the class stays undecided, nothing is called infeasible
    (HiGHS solves all 12), and the controls are solved at HiGHS's optimum."""
    d = np.load(FIXTURE)
    num, ctrl = _kinds(d, len(replay))
    executed = 0
    assert len(num) == 12 and len(ctrl) == 4
    executed += 1
    decided = sum(replay[i]["status"] in ("optimal", "infeasible") for i in num)
    assert decided <= 2, (
        f"{decided}/12 decided: the warm dual now decides the #2142 class; move the "
        "strict xfail below to a pass and update this pin"
    )
    executed += 1
    for i in num:
        assert replay[i]["status"] != "infeasible", f"lp{i}: false infeasible"
        executed += 1
    for i in ctrl:
        ref = float(d[f"lp{i}_highs_obj"])
        assert replay[i]["status"] == "optimal"
        assert abs(replay[i]["obj"] - ref) <= 1e-6 * (1.0 + abs(ref))
        assert _audit_ok(d, i, np.asarray(replay[i]["x"]))
        executed += 3
    print(f"#2142 node-LP replay: decided {decided}/12; executed assertions: {executed}")
    assert executed == 1 + 1 + 12 + 3 * 4


@pytest.mark.xfail(
    strict=True,
    reason="#2142: the warm dual leaves these node LPs undecided "
    "(DISCOPT_LP_DUAL_REBUILD, which decided them, was retired by its panel)",
)
def test_cstr_bigm_node_lps_decided_at_highs_optimum(replay):
    """The bar a fix must meet: at least 10 of 12 ``Optimal`` at HiGHS's objective
    on a point the original-LP audit accepts, with row duals for the NS bound."""
    d = np.load(FIXTURE)
    num, _ = _kinds(d, len(replay))
    on_opt = [i for i in num if replay[i]["status"] == "optimal"]
    assert len(on_opt) >= 10, f"decided {len(on_opt)}/12"
    for i in on_opt:
        ref = float(d[f"lp{i}_highs_obj"])
        assert abs(replay[i]["obj"] - ref) <= 1e-6 * (1.0 + abs(ref))
        assert _audit_ok(d, i, np.asarray(replay[i]["x"]))
        assert replay[i]["n_dual"] == int(d[f"lp{i}_m"])
