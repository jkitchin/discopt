"""#1013 tree-level differential panel: one LP-engine flag ON vs OFF through B&B.

The LP panel measures one node LP; this measures whole solves, which is where
CLAUDE.md §5's cert-clean bar lives: no bound may pass its reference optimum, no
incumbent may beat it, no `optimal` may certify a wrong value, and no instance
that certified may stop certifying.

Instances are every vendored `.nl` with a recorded optimum in
`python/tests/data/known_optima.toml` (no hardcoded names, §2), capped by
`I1013_N`. Arms run in child processes (the flag is read once per process),
interleaved per instance.

The flag under test is `I1013_FLAG` (default `DISCOPT_LP_DUAL_STALL_BAIL`, the
bail this harness was written for; `DISCOPT_LP_DUAL_COST_PERTURB` is the other
#1013 mechanism). Both arms assert the loaded build actually carries the flag's
counters before anything is recorded from it (CLAUDE.md §8).

Prints a checked-assertion count and exits non-zero at zero (§6).
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python", "tests"))
from _optima import known_optimum  # noqa: E402

DATA = os.path.join(ROOT, "python/tests/data/minlplib_nl")
FLAG = os.environ.get("I1013_FLAG", "DISCOPT_LP_DUAL_STALL_BAIL")
# The counter that proves the build under test carries the flag's mechanism. A
# panel run against a stale extension would otherwise produce two identical arms
# and read as "bound-neutral" (CLAUDE.md §8).
MARKER = {
    "DISCOPT_LP_DUAL_STALL_BAIL": "DualDegenerateStallBails",
    "DISCOPT_LP_DUAL_COST_PERTURB": "DualCostPerturbAttempts",
}[FLAG]
TL = float(os.environ.get("I1013_TL", "60"))
N = int(os.environ.get("I1013_N", "14"))

CHILD = r"""
import json, sys
import discopt._rust as _rust
import discopt.modeling as dm
assert sys.argv[3] in dict(_rust.profile_counters_py()), (
    f"stale discopt._rust at {_rust.__file__}: no {sys.argv[3]} counter, so this "
    f"build predates the mechanism under test"
)
from discopt.modeling.core import ObjectiveSense
m = dm.from_nl(sys.argv[1])
r = m.solve(time_limit=float(sys.argv[2]))
print('OUT ' + json.dumps({
    'status': r.status, 'obj': r.objective, 'bound': r.bound,
    'nodes': r.node_count, 'gap_certified': bool(r.gap_certified),
    'wall': r.wall_time,
    'maximize': m._objective.sense == ObjectiveSense.MAXIMIZE,
}))
"""


def run(path, arm):
    env = dict(os.environ)
    env[FLAG] = arm
    p = subprocess.run(
        [sys.executable, "-u", "-c", CHILD, path, str(TL), MARKER],
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0, f"{path} arm={arm}: {p.stderr[-2500:]}"
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("OUT ")]
    assert len(lines) == 1, f"{path} arm={arm}: {len(lines)} OUT lines"
    return json.loads(lines[0][4:])


names = []
for f in sorted(os.listdir(DATA)):
    if not f.endswith(".nl"):
        continue
    try:
        known_optimum(f[:-3])
    except KeyError:
        continue
    names.append(f[:-3])
names = names[:N]
print(
    f"instances with a recorded optimum: {len(names)}  time limit {TL}s  flag {FLAG}",
    flush=True,
)

checks = 0
issues = []
for nm in names:
    opt = known_optimum(nm)
    res = {arm: run(os.path.join(DATA, nm + ".nl"), arm) for arm in ("0", "1")}
    off, on = res["0"], res["1"]
    tol = 1e-4 * (1.0 + abs(opt))
    for arm, r in res.items():
        if r["bound"] is not None and abs(r["bound"]) != float("inf"):
            bad = (r["bound"] < opt - tol) if r["maximize"] else (r["bound"] > opt + tol)
            checks += 1
            if bad:
                issues.append(f"{nm} arm={arm}: UNSOUND bound {r['bound']} vs optimum {opt}")
        if r["obj"] is not None:
            bad = (r["obj"] > opt + tol) if r["maximize"] else (r["obj"] < opt - tol)
            checks += 1
            if bad:
                issues.append(f"{nm} arm={arm}: incumbent {r['obj']} beyond optimum {opt}")
    checks += 1
    if off["gap_certified"] and not on["gap_certified"]:
        issues.append(f"{nm}: certification REGRESSED (off certified, on did not)")
    checks += 1
    if off["status"] == "optimal" and on["status"] != "optimal":
        issues.append(f"{nm}: status regressed {off['status']} -> {on['status']}")
    print(
        f"{nm:20s} off {off['status']:10s} obj={off['obj']!r} bound={off['bound']!r} "
        f"nodes={off['nodes']} {off['wall']:.1f}s | on {on['status']:10s} obj={on['obj']!r} "
        f"bound={on['bound']!r} nodes={on['nodes']} {on['wall']:.1f}s",
        flush=True,
    )

print(f"\nassertions checked: {checks}")
print(f"issues: {len(issues)}")
for i in issues:
    print("  ", i)
if checks == 0 or issues:
    sys.exit(1)
