"""End to end: the default-ON Rust MILP sub-MIP heuristic (``DISCOPT_MILP_SUBMIP``).

PR #2196 review: deleting the RENS/RINS unit files left no test that the sub-MIP, a
default path since highs-parity D2, actually runs on a real solve, never has a proposed
incumbent rejected by the base-row check, and leaves the certified answer what the
exhaustive oracle says. Each arm runs in its own process: the profile counters are
process-global and are only collected under ``DISCOPT_PROFILE=1``.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

N, K, SEED = 20, 5, 0

_PROBE = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
import discopt.modeling as dm
import discopt._rust as R

n, k, seed = (int(v) for v in sys.argv[1:4])
rng = np.random.default_rng(seed)
W = rng.integers(5, 40, (k, n))
v = rng.integers(10, 50, n)
m = dm.Model("mk")
x = m.binary("x", shape=(n,))
m.maximize(sum(int(v[i]) * x[i] for i in range(n)))
for r in range(k):
    m.subject_to(sum(int(W[r, i]) * x[i] for i in range(n)) <= int(W[r].sum() // 2))
R.profile_reset_py()
res = m.solve(time_limit=120, milp_backend="native")
ctr = {a: int(b) for a, b in R.profile_counters_py().items() if a.startswith("SubMip")}
print("ROW " + json.dumps({"status": res.status, "cert": bool(res.gap_certified),
                           "obj": res.objective, "bound": res.bound, "ctr": ctr}))
"""


def _run(submip: str) -> dict:
    env = dict(os.environ, DISCOPT_PROFILE="1", DISCOPT_MILP_SUBMIP=submip)
    p = subprocess.run(
        [sys.executable, "-c", _PROBE, str(N), str(K), str(SEED)],
        env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
    assert p.returncode == 0 and len(rows) == 1, p.stderr[-2000:]
    return json.loads(rows[0][4:])


def _oracle() -> float:
    rng = np.random.default_rng(SEED)
    W = rng.integers(5, 40, (K, N))
    v = rng.integers(10, 50, N)
    cap = W.sum(axis=1) // 2
    pts = np.array(list(itertools.product((0, 1), repeat=N)), dtype=np.int64)
    feas = (pts @ W.T <= cap).all(axis=1)
    return float((pts[feas] @ v).max())


def test_the_submip_runs_is_never_rejected_and_keeps_the_certified_optimum():
    on, off = _run("1"), _run("0")
    want = _oracle()
    ctr = on["ctr"]
    assert ctr.get("SubMipRensRun", 0) + ctr.get("SubMipRinsRun", 0) >= 1, ctr  # it ran (§6)
    assert ctr.get("SubMipRejected", 0) == 0, ctr  # no proposed point failed the base rows
    assert off["ctr"].get("SubMipConsidered", 0) == 0, off["ctr"]  # the opt-out is off
    for arm in (on, off):
        assert arm["status"] == "optimal" and arm["cert"], arm
        assert arm["obj"] == pytest.approx(want, abs=1e-6), (arm["obj"], want)
        assert arm["bound"] >= want - 1e-6  # maximize: the bound never undercuts the optimum
