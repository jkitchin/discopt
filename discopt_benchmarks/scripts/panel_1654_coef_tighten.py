"""#1654 graduation panel: DISCOPT_MILP_COEF_TIGHTEN ON vs OFF on big-M MILP families.

Usage: python -u discopt_benchmarks/scripts/panel_1654_coef_tighten.py OUT.jsonl [n_seeds]

Each family is the same MILP for every M at or above its activity bound, so the
reference optimum is the M_ref solve. Arms are interleaved and every solve runs in a
fresh subprocess so the flag is read cleanly. Exits non-zero when no oracle check ran
(CLAUDE.md section 6). Results: docs/dev/issue-1654-coef-tighten-panel-2026-10-05.md.
"""

import collections
import itertools
import json
import os
import subprocess
import sys

import discopt.modeling as dm
import numpy as np


def sequencing(seed, M):
    rng = np.random.default_rng(seed)
    n = 6
    p = rng.integers(2, 10, n)
    r = rng.integers(0, 9, n)
    m = dm.Model(f"seq{seed}")
    s = [m.continuous(f"s{i}", lb=float(r[i]), ub=100) for i in range(n)]
    C = m.continuous("C", lb=0, ub=100)
    for i, j in itertools.combinations(range(n), 2):
        y = m.binary(f"y{i}_{j}")
        m.subject_to(s[i] + float(p[i]) <= s[j] + M * (1 - y))
        m.subject_to(s[j] + float(p[j]) <= s[i] + M * y)
    for i in range(n):
        m.subject_to(s[i] + float(p[i]) <= C)
    m.minimize(C)
    return m


sequencing.M_ref = 200.0


def facility(seed, M):
    rng = np.random.default_rng(seed)
    nf, nc = 4, 6
    cost = rng.uniform(1, 10, (nf, nc))
    fc = rng.uniform(5, 30, nf)
    dem = rng.uniform(1, 5, nc)
    cap = rng.uniform(8, 15, nf)
    m = dm.Model(f"fac{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{c}", lb=0, ub=10) for c in range(nc)] for i in range(nf)]
    for c in range(nc):
        m.subject_to(sum(x[i][c] for i in range(nf)) >= float(dem[c]))
    for i in range(nf):
        m.subject_to(sum(x[i][c] for c in range(nc)) <= float(cap[i]))
        for c in range(nc):
            m.subject_to(x[i][c] <= M * y[i])
    m.minimize(
        sum(float(cost[i, c]) * x[i][c] for i in range(nf) for c in range(nc))
        + sum(float(fc[i]) * y[i] for i in range(nf))
    )
    return m


facility.M_ref = 10.0


def fixed_charge(seed, M):
    rng = np.random.default_rng(seed)
    n = 8
    f = rng.uniform(10, 100, n)
    v = rng.uniform(1, 30, n)
    ub = rng.uniform(3, 12, n)
    D = float(rng.uniform(0.4, 0.8) * ub.sum())
    m = dm.Model(f"fc{seed}")
    x = [m.continuous(f"x{i}", lb=0, ub=float(ub[i])) for i in range(n)]
    y = [m.binary(f"y{i}") for i in range(n)]
    m.subject_to(sum(x) >= D)
    for i in range(n):
        m.subject_to(x[i] <= M * y[i])
    m.minimize(sum(float(f[i]) * y[i] + float(v[i]) * x[i] for i in range(n)))
    return m


fixed_charge.M_ref = 12.0

FAMILIES = {"sequencing": sequencing, "facility": facility, "fixed_charge": fixed_charge}



HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = r"""
import json, sys, time, warnings, contextlib, io
warnings.simplefilter("ignore")
sys.path.insert(0, HERE)
import discopt, discopt.solvers.lp_milp_highs as L
assert hasattr(L, "coefficient_tightened"), "branch marker missing"
from panel_1654_coef_tighten import FAMILIES
fam, seed, M = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
m = FAMILIES[fam](seed, M)
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    r = m.solve(time_limit=60)
wall = time.perf_counter() - t
viol = None
if r.x is not None:
    # independent feasibility check on the model's own constraints
    import numpy as np
    from discopt.solver import _make_evaluator, _infer_constraint_bounds, _unpack_solution
    ev = _make_evaluator(m); cl, cu = _infer_constraint_bounds(m, ev)
    x = np.concatenate([np.atleast_1d(np.asarray(r.x[v.name], float)).ravel() for v in m._variables])
    g = np.asarray(ev.evaluate_constraints(x), float)
    viol = float(np.max(np.maximum(np.maximum(np.asarray(cl) - g, g - np.asarray(cu)), 0))) if len(cl) else 0.0
    ints = [x[i] for i,(v) in enumerate([vv for vv in m._variables for _ in range(vv.size)]) if v.var_type.name in ("BINARY","INTEGER")]
    viol = max(viol, max([abs(a-round(a)) for a in ints] or [0.0]))
print(json.dumps(dict(status=r.status, obj=r.objective, bound=r.bound, cert=bool(r.gap_certified),
    wall=wall, viol=viol, route=(r.algorithm_route or "")[:200], file=discopt.__file__)))
""".replace("HERE", repr(HERE))


def run(fam, seed, M, flag):
    env = dict(os.environ, DISCOPT_MILP_COEF_TIGHTEN=flag)
    p = subprocess.run(
        [sys.executable, "-c", CHILD, fam, str(seed), repr(M)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        return dict(status="CRASH", err=p.stderr[-500:])
    return json.loads(line[-1])


def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    TOL = lambda v: 1e-6 * max(1.0, abs(v))
    bad = []
    stats = collections.defaultdict(collections.Counter)
    walls = collections.defaultdict(list)
    checks = 0
    for r in rows:
        f = r["flag"]
        ref = r["ref"]
        stats[f]["n"] += 1
        stats[f][r["status"]] += 1
        if r.get("cert"):
            stats[f]["certified"] += 1
        walls[f].append(r.get("wall", 0.0))
        if ref is None or r["status"] == "CRASH":
            if r["status"] == "CRASH":
                bad.append(("crash", r))
            continue
        checks += 1
        if r.get("bound") is not None and r["bound"] > ref + TOL(ref):
            bad.append(("bound above reference", r))
        if r.get("obj") is not None:
            if r.get("viol") is None or r["viol"] > 1e-5:
                bad.append(("incumbent infeasible", r))
            if r["obj"] < ref - TOL(ref) and r.get("viol", 1) <= 1e-5:
                bad.append(("objective below reference", r))
        if r.get("cert") and (
            r.get("obj") is None or abs(r["obj"] - ref) > 1e-4 * max(1, abs(ref))
        ):
            if r["status"] == "optimal":
                bad.append(("certified wrong value", r))
    # certification regressions: an (fam, seed, M) certified OFF but not ON
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["fam"], r["seed"], r["M"])][r["flag"]] = r
    reg = [
        k for k, d in by.items() if d.get("0", {}).get("cert") and not d.get("1", {}).get("cert")
    ]
    gain = [
        k for k, d in by.items() if d.get("1", {}).get("cert") and not d.get("0", {}).get("cert")
    ]
    import statistics as st

    for f in ("0", "1"):
        w = walls[f]
        print(f"flag={f}: {dict(stats[f])}  wall total {sum(w):.1f}s median {st.median(w):.2f}s")
    print("certification regressions (ON loses a cert):", reg)
    print("certification gains:", len(gain))
    print("soundness violations:", len(bad))
    for why, r in bad[:20]:
        print(
            "  ",
            why,
            {
                k: r.get(k)
                for k in ("fam", "seed", "M", "flag", "status", "obj", "bound", "ref", "viol")
            },
        )
    print("executed oracle checks:", checks)
    return checks


if __name__ == "__main__":
    out = sys.argv[1]
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(4)
    Ms = [1e4, 1e6, 1e8, 1e10]
    rows = []
    n_done = 0
    with open(out, "w") as fh:
        for fam, b in FAMILIES.items():
            for seed in seeds:
                ref = run(fam, seed, b.M_ref, "0")
                for M in Ms:
                    for k, flag in enumerate(
                        ("0", "1") if (seed + int(M) % 7) % 2 == 0 else ("1", "0")
                    ):
                        res = run(fam, seed, M, flag)
                        rec = dict(
                            fam=fam,
                            seed=seed,
                            M=M,
                            flag=flag,
                            ref=ref.get("obj"),
                            ref_status=ref.get("status"),
                            ref_cert=ref.get("cert"),
                            **res,
                        )
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_done += 1
                        print(
                            n_done,
                            fam,
                            seed,
                            f"{M:.0e}",
                            flag,
                            res.get("status"),
                            res.get("obj"),
                            res.get("bound"),
                            res.get("cert"),
                            f"{res.get('wall', 0):.1f}s",
                            flush=True,
                        )
    print("executed solves:", n_done)
    sys.exit(0 if analyze(out) else 1)
