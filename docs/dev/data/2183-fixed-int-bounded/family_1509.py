"""#2183 substitute for the #1509 family panel (scratchpad/i1509/panel.py is not in the repo).

Rebuilt from test_1509's generators and the panel description in lp-milp-highs-routing-plan.md:
hand-written PWL MILPs (``_pwl_block``) over c in {1e2..1e5}, K in {4,8,12,16}, log band in
{8.67e-8, 5e-7, 2e-6}, sq band taper on/off, plus nonlinear_to_pwl surrogates of log/sqrt at
3 segment counts. Oracle: c**2 (x=c, n=0 feasible; surrogates relax that model). Arms:
``new`` = this branch; ``old`` = _fixed_integer_lp_cannot_refute patched to False, which is
exactly the pre-#2183 code path. Interleaved per instance.
"""
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[4])
import json, math, os, sys
os.environ["DISCOPT_LP_MILP_BACKEND"] = "highs"
import numpy as np
import discopt, discopt.modeling as dm
assert discopt.__file__.startswith(REPO + "/python"), discopt.__file__
from discopt.solvers import lp_milp_highs as L
assert hasattr(L, "_fixed_integer_lp_cannot_refute")
sys.path.insert(0, REPO + "/python/tests")
from test_1509_highs_milp_refuted_certificate import _pwl_block

REAL = L._fixed_integer_lp_cannot_refute
fired = {"n": 0, "true": 0}
def spy(*a, **k):
    r = REAL(*a, **k); fired["n"] += 1; fired["true"] += bool(r); return r

def direct(c, K, band, taper):
    m = dm.Model("fam")
    x = m.continuous("x", lb=c, ub=c + 10)
    n = m.integer("n", lb=0, ub=1e6)
    sqband = (10.0 / K) ** 2 / 4 if taper else 1.5625
    sq = _pwl_block(m, x, c, K, lambda t: t * t, sqband, "sq")
    lg = _pwl_block(m, x, c, K, lambda t: math.log(t + 1), band, "lg")
    m.minimize(sq + n); m.subject_to(lg + n >= 2)
    return m

def surrogate(c, f, seg):
    m = dm.Model("sur")
    x = m.continuous("x", lb=c, ub=c + 10); n = m.integer("n")
    m.minimize(x**2 + n)
    m.subject_to((dm.log(x + 1) if f == "log" else dm.sqrt(x + 1)) + n >= 2)
    return dm.nonlinear_to_pwl(m, segments=seg)

cases = []
for c in (1e2, 1e3, 1e4, 3e4, 1e5):
    for K in (4, 8, 12, 16):
        for band in (8.67e-8, 5e-7, 2e-6):
            for taper in (False, True):
                cases.append(("direct", (c, K, band, taper)))
for c in (1e2, 1e3, 3e3, 1e4, 3e4, 1e5):
    for f in ("log", "sqrt"):
        for seg in (4, 8, 16):
            cases.append(("pwl", (c, f, seg)))
print("cases", len(cases), flush=True)
rows = []; cmp = 0
for i, (kind, p) in enumerate(cases):
    row = {"kind": kind, "p": list(p)}
    for arm in ("old", "new"):
        L._fixed_integer_lp_cannot_refute = (lambda *a, **k: False) if arm == "old" else spy
        before = fired["true"]
        m = direct(*p) if kind == "direct" else surrogate(*p)
        r = m.solve(time_limit=60)
        ref = p[0] ** 2
        false_cert = bool(r.gap_certified and r.bound is not None and r.bound > ref + 1e-6 * (1 + abs(ref)))
        row[arm] = dict(status=r.status, cert=bool(r.gap_certified), obj=r.objective, bound=r.bound,
                        nodes=r.node_count, false_cert=false_cert, bounded_fired=fired["true"] - before)
        cmp += 1
    L._fixed_integer_lp_cannot_refute = REAL
    rows.append(row)
    o, n = row["old"], row["new"]
    same = (o["status"], o["cert"], o["obj"], o["bound"]) == (n["status"], n["cert"], n["obj"], n["bound"])
    print(f"[{i+1}/{len(cases)}] {kind} {p} old={o['status']}/{o['cert']}/fc={o['false_cert']} "
          f"new={n['status']}/{n['cert']}/fc={n['false_cert']} fired={n['bounded_fired']} same={same}", flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
fc_old = sum(r["old"]["false_cert"] for r in rows); fc_new = sum(r["new"]["false_cert"] for r in rows)
cert_old = sum(r["old"]["cert"] for r in rows); cert_new = sum(r["new"]["cert"] for r in rows)
diff = sum((r["old"]["status"], r["old"]["cert"], r["old"]["obj"], r["old"]["bound"]) !=
           (r["new"]["status"], r["new"]["cert"], r["new"]["obj"], r["new"]["bound"]) for r in rows)
print(f"SUMMARY n={len(rows)} comparisons={cmp} false_cert old={fc_old} new={fc_new} "
      f"certified old={cert_old} new={cert_new} differing={diff} bounded_arm_fired={fired['true']}/{fired['n']} calls")
sys.exit(0 if cmp else 1)
