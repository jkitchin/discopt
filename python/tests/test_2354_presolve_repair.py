"""#2354: a postsolved MILP point that misses an original row is repaired, not discarded.

The MILP driver solves a presolved (reduced) problem, postsolves its incumbent and
re-checks it on the original rows at 1e-6. Before #2354 a point that failed that
check threw the whole reduced search away and the original problem was solved cold.
Now the integers are fixed and the original LP is re-solved for the continuous columns
(HiGHS's ``transformNewIntegerFeasibleSolution`` repair). The repaired point must pass
the same, unchanged check.

Fixture: a captured OA master of ``portfol_roundlot`` (MINLPLib). Lot rows pin each
holding at ``n_k / lot_k``, and the reduced search's point misses a lot row by
1.8e-3, which is far outside the 1e-6 check. The exact fixed-integer LP is infeasible:
the budget row misses ``= 1`` by 2.3e-8, so only a completion within tolerance exists.
Measured on ``origin/main`` 16e678f2d: the original was re-solved cold (1373 nodes)
after the reduced search, and the returned objective was 0.028290434.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np

DATA = pathlib.Path(__file__).parent / "data" / "milp_2354_presolve_repair.npz"

_PROBE = r"""
import sys, numpy as np
import discopt._rust as R
d = np.load(sys.argv[1])
a = [d[f"a{i}"] for i in range(14)]
args = [a[0], int(a[1]), int(a[2]), a[3], a[4], a[5], a[6], a[7], a[8], a[9], int(a[10]),
        float(a[11]), int(a[12]), float(a[13])]
out = R.solve_milp_csc_py(*args, root_cuts=int(d["root_cuts"]), cut_rounds=int(d["cut_rounds"]),
                          cut_select=True, cmir_cuts=True, time_limit_s=120.0)
np.save(sys.argv[2], np.asarray(out[1], dtype=float))
print("RESULT", out[0], repr(out[2]), repr(out[3]), out[4], flush=True)
ctr = dict(R.profile_counters_py())
print("COUNTERS", ctr["MilpPresolveRepairAttempts"], ctr["MilpPresolveRepaired"],
      ctr["MilpPresolveRepairSeeded"], flush=True)
"""


def test_postsolved_point_is_repaired_and_verifies(tmp_path):
    xfile = tmp_path / "x.npy"
    env = dict(os.environ, DISCOPT_MILP_PRESOLVE_LOG="1", RAYON_NUM_THREADS="1")
    # The repair counters accumulate only under DISCOPT_PROFILE; the probe reads the
    # cumulative totals (`profile_counters_py`), which the end-of-solve dump leaves intact.
    env["DISCOPT_PROFILE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(DATA), str(xfile)],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    log = proc.stderr
    # The fixture must still reach the path under test (CLAUDE.md section 6). If a
    # presolve change stops it from failing verification, this test no longer
    # exercises the repair and has to be re-captured.
    assert "postsolved point failed verification" in log, log[-4000:]
    assert "repaired by the fixed-integer LP" in log, log[-4000:]
    assert "failed verification; solving the original" not in log
    assert "repair did not verify" not in log
    # The instrument must see the path the log shows (section 6): a counter reading 0
    # here would let a probe report "no repairs" on a run that made one.
    ctr = next(s for s in proc.stdout.splitlines() if s.startswith("COUNTERS"))
    attempts, repaired, seeded = (int(v) for v in ctr.split()[1:])
    assert attempts >= 1 and repaired >= 1, ctr
    assert seeded == 0, ctr

    line = next(s for s in proc.stdout.splitlines() if s.startswith("RESULT"))
    _, status, obj, bound, _nodes = line.split()
    obj, bound = float(obj), float(bound)
    assert status == "optimal", line
    assert bound <= obj, line

    # The returned point passes the original-model check: equality rows of the
    # standard form, the box, integrality and the reported objective.
    d = np.load(DATA)
    c, m, n = d["a0"], int(d["a1"]), int(d["a2"])
    col_ptr, row_idx, vals = d["a3"], d["a4"], d["a5"]
    b, lb, ub, ints, ns = d["a6"], d["a7"], d["a8"], d["a9"], int(d["a10"])
    x = np.load(xfile)
    assert x.shape == (n,) or x.shape == (ns,), x.shape
    xs = np.zeros(n)
    xs[: x.shape[0]] = x
    checked = 0
    act = np.zeros(m)
    for j in range(ns):
        for k in range(col_ptr[j], col_ptr[j + 1]):
            act[row_idx[k]] += vals[k] * xs[j]
    # Row i is `act_i + s_i = b_i` with its slack column(s) boxed: the implied slack
    # value must lie in the slack's box, within the 1e-6 relative check.
    slack_of = {}
    for j in range(ns, n):
        for k in range(col_ptr[j], col_ptr[j + 1]):
            slack_of.setdefault(row_idx[k], []).append((j, vals[k]))
    for i in range(m):
        r = b[i] - act[i]
        tol = 1e-6 * (1.0 + abs(act[i]))
        cols = slack_of.get(i, [])
        if not cols:
            assert abs(r) <= tol, (i, r)
        else:
            assert len(cols) == 1, (i, cols)
            j, a = cols[0]
            lo, hi = sorted((a * lb[j], a * ub[j]))
            assert lo - tol <= r <= hi + tol, (i, r, lo, hi)
        checked += 1
    for j in range(ns):
        t = 1e-6 * (1.0 + abs(xs[j]))
        assert lb[j] - t <= xs[j] <= ub[j] + t, (j, xs[j])
        checked += 1
    for j in ints:
        assert abs(xs[j] - round(xs[j])) <= 1e-5, (j, xs[j])
        checked += 1
    assert abs(float(c[:ns] @ xs[:ns]) + float(d["a11"]) - obj) <= 1e-9 * (1 + abs(obj))
    assert checked == m + ns + len(ints) and checked > 0
