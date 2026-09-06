"""#1013 tree-level differential panel over the vendored QPLIB instances.

`tree_panel.py` runs the MINLPLib `.nl` instances that carry a recorded optimum
— 16 of them, and none is in the degenerate-lifted-relaxation class this issue is
about, so it can only show *no harm*. These 9 QPLIB instances are that class:
they are where the LP-level panel's biggest pivot reductions live.

Oracle is `qplib.solu` (`=best=` per instance), read through the reader rather
than parsed here. Gates, per instance and arm:

  * the dual bound may not pass the reference optimum (the unsound direction),
  * an incumbent may not beat it,
  * an instance that certified its gap with the flag OFF must still certify ON.

Instances are every `.qplib` under the vendored corpus with an entry in the
`.solu` — selected by filter, never by name (CLAUDE.md §2). Both arms assert the
loaded build carries the flag's counter before anything is recorded from it (§8),
and the ON arm reports the perturbation counters so a run where the mechanism
never fired cannot be read as evidence about it (§6).

    I1013_TL=120 python -u scratchpad/i1013/qplib_tree_panel.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "python/tests/data/qplib")
QDIR = os.path.join(DATA, "qplib")
TL = float(os.environ.get("I1013_TL", "120"))
FLAG = "DISCOPT_LP_DUAL_COST_PERTURB"
MARKER = "DualCostPerturbAttempts"

CHILD = r"""
import json, sys
import discopt._rust as _rust
from discopt.interfaces.qplib import from_qplib, read_qplib
snap = dict(_rust.profile_counters_py())
assert sys.argv[3] in snap, (
    f"stale discopt._rust at {_rust.__file__}: no {sys.argv[3]} counter, so this "
    f"build predates the mechanism under test"
)
inst = read_qplib(sys.argv[1])
m = from_qplib(sys.argv[1])
r = m.solve(time_limit=float(sys.argv[2]))
c = dict(_rust.profile_counters_py())
print('OUT ' + json.dumps({
    'status': r.status, 'obj': r.objective, 'bound': r.bound,
    'nodes': r.node_count, 'gap_certified': bool(r.gap_certified),
    'wall': r.wall_time,
    'attempts': c[sys.argv[3]], 'accepted': c['DualCostPerturbAccepted'],
    'maximize': inst.is_maximize(),
}))
"""


def run(path, arm):
    env = dict(os.environ)
    env[FLAG] = arm
    env["DISCOPT_PROFILE"] = "1"
    p = subprocess.run(
        [sys.executable, "-u", "-c", CHILD, path, str(TL), MARKER],
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0, f"{path} arm={arm}: {p.stderr[-2500:]}"
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("OUT ")]
    assert len(lines) == 1, f"{path} arm={arm}: {len(lines)} OUT lines\n{p.stdout[-1500:]}"
    return json.loads(lines[0][4:])


sys.path.insert(0, os.path.join(ROOT, "python"))
from discopt.interfaces.qplib import read_solu  # noqa: E402

solu = read_solu(os.path.join(DATA, "qplib.solu"))
names = sorted(f[:-6] for f in os.listdir(QDIR) if f.endswith(".qplib") and f[:-6] in solu)
print(f"QPLIB instances with a reference optimum: {len(names)}  time limit {TL}s", flush=True)

checks = 0
issues = []
fired = 0
for nm in names:
    opt = solu[nm]
    res = {arm: run(os.path.join(QDIR, nm + ".qplib"), arm) for arm in ("0", "1")}
    off, on = res["0"], res["1"]
    if on["attempts"] >= 1:
        fired += 1
    tol = 1e-4 * (1.0 + abs(opt))
    for arm, r in res.items():
        # QPLIB carries BOTH senses (4 of the 9 vendored instances are maximize),
        # and the dual bound is an upper bound on a maximize instance. Reading the
        # sense off the instance is not optional: assuming minimize reported four
        # "UNSOUND bound" issues on this panel, in BOTH arms, on two maximize
        # instances whose bounds were perfectly correct.
        mx = r["maximize"]
        if r["bound"] is not None and abs(r["bound"]) != float("inf"):
            checks += 1
            bad = (r["bound"] < opt - tol) if mx else (r["bound"] > opt + tol)
            if bad:
                issues.append(f"{nm} arm={arm}: UNSOUND bound {r['bound']} vs optimum {opt}")
        if r["obj"] is not None:
            checks += 1
            bad = (r["obj"] > opt + tol) if mx else (r["obj"] < opt - tol)
            if bad:
                issues.append(f"{nm} arm={arm}: incumbent {r['obj']} beyond optimum {opt}")
    checks += 1
    if off["gap_certified"] and not on["gap_certified"]:
        issues.append(f"{nm}: certification REGRESSED")
    checks += 1
    if off["status"] == "optimal" and on["status"] != "optimal":
        issues.append(f"{nm}: status regressed {off['status']} -> {on['status']}")
    tighter = "same"
    if off["bound"] is not None and on["bound"] is not None:
        d = on["bound"] - off["bound"]
        if abs(d) > 1e-9 * (1.0 + abs(off["bound"])):
            tighter = "ON" if ((d < 0) if on["maximize"] else (d > 0)) else "OFF"
    print(
        f"{nm:14s} {'max' if on['maximize'] else 'min'} "
        f"off {off['status']:10s} bound={off['bound']!r} nodes={off['nodes']} "
        f"{off['wall']:.1f}s | on {on['status']:10s} bound={on['bound']!r} "
        f"nodes={on['nodes']} {on['wall']:.1f}s  fired={on['attempts']}/{on['accepted']} "
        f"tighter={tighter}",
        flush=True,
    )

wall_off = wall_on = 0.0
print(f"\nassertions checked: {checks}")
print(f"instances where the perturbation fired: {fired} / {len(names)}")
print(f"issues: {len(issues)}")
for i in issues:
    print("  ", i)
if checks == 0 or issues:
    sys.exit(1)
