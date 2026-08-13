"""#1008 A/B: `DISCOPT_LU_SYMBOLIC_REUSE` OFF vs ON, interleaved (CLAUDE.md #9).

Each LP is run OFF, ON, OFF, ON, ... `REPS` times each, alternating so a drift in
machine load hits both arms equally, and the spread is reported. The arms run in
CHILD processes because the flag is read once per process via `OnceLock`.

Correctness is checked on every pair: the ON arm's objective must match the OFF
arm's to 1e-9 relative and its status must not regress from `optimal`. A speedup
bought with a different answer is not a speedup (CLAUDE.md §1).

Prints a compared count and exits non-zero at zero (#6). Nothing is caught (#7).
"""

import glob
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
assert "DISCOPT_LU_SYMBOLIC_REUSE" in strs, "loaded a build without the #1008 reuse flag"

REPS = int(os.environ.get("I1008_REPS", "3"))
TL = float(os.environ.get("I1008_TL", "45"))

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
                           'facs': snap.get('LuSparseFactorizations', 0),
                           'reused': snap.get('LuSymbolicReused', 0),
                           'refill': snap.get('LuSymbolicRefreshFill', 0),
                           'refail': snap.get('LuSymbolicRefreshFail', 0)}}))
"""


def run(path, on):
    env = dict(os.environ)
    env["DISCOPT_LU_SYMBOLIC_REUSE"] = "1" if on else "0"
    # Counters (not the phase dump) are all this needs, and DISCOPT_PROFILE also
    # gates them; the dump goes to stderr and is not parsed here.
    env["DISCOPT_PROFILE"] = "1"
    code = CHILD.format(wtpy=WT + "/python", path=path, tl=TL)
    p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0, f"{path} on={on}: rc={p.returncode}\n{p.stderr[-3000:]}"
    res = [json.loads(ln[4:]) for ln in p.stdout.splitlines() if ln.startswith("RES ")]
    assert len(res) == 1, f"{path} on={on}: expected 1 RES line, got {len(res)}"
    return res[0]


print(subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), flush=True)
paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
assert paths, "no captured LPs"
print(
    f"{'tag':22s} {'off mean+-sd':>18s} {'on mean+-sd':>18s} {'speedup':>8s} "
    f"{'facs':>5s} {'reuse%':>6s} {'refill':>6s}",
    flush=True,
)
compared = 0
ratios = []
violations = []
checks = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    offs, ons, last_off, last_on = [], [], None, None
    for _ in range(REPS):
        r0 = run(p, False)
        offs.append(r0["wall"])
        last_off = r0
        r1 = run(p, True)
        ons.append(r1["wall"])
        last_on = r1
    assert last_off["reused"] == 0, f"{tag}: the OFF arm reused an ordering"
    # Correctness gate on every pair. Violations are COLLECTED, printed as they
    # happen, and turned into a non-zero exit at the end -- aborting on the first
    # would leave the other 17 instances unmeasured, and hiding one would be #7.
    checks += 1
    if last_off["status"] == "optimal":
        if last_on["status"] != "optimal":
            v = f"{tag}: status regressed {last_off['status']} -> {last_on['status']}"
            print("VIOLATION " + v, flush=True)
            violations.append(v)
        else:
            d = abs(last_on["obj"] - last_off["obj"]) / max(1.0, abs(last_off["obj"]))
            if d >= 1e-9:
                v = f"{tag}: objective drift {d:.3e} ({last_off['obj']} vs {last_on['obj']})"
                print("VIOLATION " + v, flush=True)
                violations.append(v)
    mo, mn = statistics.mean(offs), statistics.mean(ons)
    so = statistics.stdev(offs) if len(offs) > 1 else 0.0
    sn = statistics.stdev(ons) if len(ons) > 1 else 0.0
    ratios.append(mo / mn)
    facs = last_on["facs"]
    print(
        f"{tag:22s} {mo:11.3f}+-{so:5.3f} {mn:11.3f}+-{sn:5.3f} {mo / mn:7.2f}x "
        f"{facs:5d} {100 * last_on['reused'] / max(1, facs):5.1f}% {last_on['refill']:6d}",
        flush=True,
    )
    print(
        "JSONAB "
        + json.dumps(
            {
                "tag": tag,
                "off": offs,
                "on": ons,
                "speedup": mo / mn,
                "status_off": last_off["status"],
                "status_on": last_on["status"],
                "obj_off": last_off["obj"],
                "obj_on": last_on["obj"],
                "iters_off": last_off["iters"],
                "iters_on": last_on["iters"],
                "facs": facs,
                "reused": last_on["reused"],
                "refill": last_on["refill"],
                "refail": last_on["refail"],
            }
        ),
        flush=True,
    )
    compared += 1

ratios.sort()
print(
    f"speedup n={len(ratios)} min={ratios[0]:.2f}x median={ratios[len(ratios) // 2]:.2f}x "
    f"max={ratios[-1]:.2f}x",
    flush=True,
)
print(subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), flush=True)
print("compared LPs:", compared, " correctness checks executed:", checks, flush=True)
for v in violations:
    print("FINAL VIOLATION " + v, flush=True)
print("violations:", len(violations), flush=True)
if compared == 0 or violations:
    sys.exit(1)
