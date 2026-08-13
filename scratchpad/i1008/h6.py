"""#1008 H6 entry cell: does the dual stability pass clear the QPLIB_0911 stall?

Four arms per LP, interleaved (CLAUDE.md #9) so machine-load drift hits all arms
equally:

    base     nothing set                       (the 1279-pivot / 1.9 s reference)
    harris   DISCOPT_LP_DUAL_HARRIS=1
    sym      DISCOPT_LU_SYMBOLIC_REUSE=1       (the perturbation that stalls)
    both     harris + sym

Kill criterion 1 for H6 (pre-registered in HYPOTHESIS.md): H6 is falsified if the
`harris` arm's iteration count is not below the `base` arm's AND the `both` arm
still goes to `iter_limit`.

Arms run in CHILD processes because every flag is read once per process via
`OnceLock`. Prints a compared count and exits non-zero at zero (#6). Nothing is
caught (#7): a child that fails aborts the probe with its stderr.
"""

import json
import os
import statistics
import subprocess
import sys

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
strs = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
# §8: assert the marker unique to THIS version is present in the binary actually
# loaded. Without it a probe can silently measure the pre-H6 engine on both arms.
assert "DISCOPT_LP_DUAL_HARRIS" in strs, "loaded a build without the #1008 H6 flag"
assert "DISCOPT_LU_SYMBOLIC_REUSE" in strs, "loaded a build without the #1008 H5 flag"

REPS = int(os.environ.get("I1008_REPS", "3"))
TL = float(os.environ.get("I1008_TL", "45"))

ARMS = {
    "base": {},
    "harris": {"DISCOPT_LP_DUAL_HARRIS": "1"},
    "sym": {"DISCOPT_LU_SYMBOLIC_REUSE": "1"},
    "both": {"DISCOPT_LP_DUAL_HARRIS": "1", "DISCOPT_LU_SYMBOLIC_REUSE": "1"},
}

CHILD = r"""
import os, sys, time, json
sys.path.insert(0, {wtpy!r})
import numpy as np, scipy.sparse as sp
import discopt._rust as _rust
from discopt.solvers.milp_simplex import _dual_start_slack_basis
z = np.load({path!r})
nrow, ncol = int(z['shape'][0]), int(z['shape'][1])
A = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=(nrow, ncol))
c, b, lo, hi = z['c'], z['b'], z['lo'], z['hi']
st = _dual_start_slack_basis(c, lo, hi, nrow)
assert st is not None
af = sp.hstack([A, sp.identity(nrow, format='csc')], format='csc')
args = (np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)])), nrow, ncol + nrow,
        np.ascontiguousarray(af.indptr, dtype=np.int64),
        np.ascontiguousarray(af.indices, dtype=np.int64),
        np.ascontiguousarray(af.data, dtype=np.float64),
        np.ascontiguousarray(b),
        np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)])),
        np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)])),
        np.ascontiguousarray(st[0], dtype=np.int8),
        np.ascontiguousarray(st[1], dtype=np.int64), 1e-9, 100000, {tl!r})
t0 = time.perf_counter()
out = _rust.solve_lp_warm_csc_py(*args)
wall = time.perf_counter() - t0
snap = dict(_rust.profile_counters_py())
print('RES ' + json.dumps({{'wall': wall, 'status': out[0], 'obj': out[2],
                           'iters': int(out[3]),
                           'repivots': snap.get('DualHarrisRepivots', 0),
                           'degen': snap.get('DualDegeneratePivots', 0),
                           'stalls': snap.get('DualStallTrips', 0)}}))
"""


def run(path, arm):
    env = dict(os.environ)
    for k in ("DISCOPT_LP_DUAL_HARRIS", "DISCOPT_LU_SYMBOLIC_REUSE"):
        env.pop(k, None)
    env.update(ARMS[arm])
    env["DISCOPT_PROFILE"] = "1"
    code = CHILD.format(wtpy=WT + "/python", path=path, tl=TL)
    p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0, f"{path} arm={arm}: rc={p.returncode}\n{p.stderr[-3000:]}"
    res = [json.loads(ln[4:]) for ln in p.stdout.splitlines() if ln.startswith("RES ")]
    assert len(res) == 1, f"{path} arm={arm}: expected 1 RES line, got {len(res)}"
    return res[0]


def main():
    paths = sys.argv[1:]
    assert paths, "usage: h6.py <lp.npz> ..."
    want = os.environ.get("I1008_ARMS")
    if want:
        keep = want.split(",")
        for a in keep:
            assert a in ARMS, f"unknown arm {a!r}"
        for a in list(ARMS):
            if a not in keep:
                del ARMS[a]
    assert ARMS, "no arms selected"
    print(f"{'instance':<22}{'arm':<8}{'wall (s)':>16}{'iters':>8}{'status':>12}{'repiv':>8}")
    compared = 0
    rows = []
    for path in paths:
        tag = os.path.basename(path).replace(".npz", "")
        per = {a: [] for a in ARMS}
        for _ in range(REPS):
            for arm in ARMS:  # interleaved within the rep, not arm-major
                per[arm].append(run(path, arm))
        for arm in ARMS:
            w = [r["wall"] for r in per[arm]]
            last = per[arm][-1]
            sd = statistics.stdev(w) if len(w) > 1 else 0.0
            print(
                f"{tag:<22}{arm:<8}{statistics.mean(w):>10.3f}+-{sd:<5.3f}"
                f"{last['iters']:>8}{last['status']:>12}{last['repivots']:>8}",
                flush=True,
            )
            compared += 1
            rows.append({"tag": tag, "arm": arm, "walls": w, **last})
        print("JSONH6 " + json.dumps([r for r in rows if r["tag"] == tag]), flush=True)

    print(f"compared: {compared}", flush=True)
    if compared == 0:
        sys.exit("no arms were measured")


main()
