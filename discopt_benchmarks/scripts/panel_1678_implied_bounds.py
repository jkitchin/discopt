"""#1678 (c) graduation panel: DISCOPT_MILP_IMPLIED_BOUNDS ON vs OFF.

Usage:
    python -u discopt_benchmarks/scripts/panel_1678_implied_bounds.py OUT.jsonl [n_seeds] [ref_root]

Three arms, each with the flag OFF and ON, interleaved, with the arm order alternating:

* the three #1654 big-M families (finite boxes), which are regression controls;
* five big-M families whose continuous columns are left at the DEFAULT box, the class
  #1678 (c) is about. Facility location with demand equalities (the issue's model),
  with ``>=`` demand and capacities, and with ``>=`` demand and no capacities (no
  implied bound, a negative control). Also fixed charge and sequencing at the default
  box;
* the LP/MILP check instances of ``discopt_benchmarks/data/lp_milp/manifest.json``
  (HiGHS check set + i1183), loaded with ``lp_milp_loader``, oracle from the manifest.

A family is the same MILP for every M at or above its activity bound, so its oracle is
the flag-OFF solve at the family's small ``M_ref``. Every solve runs in a fresh
subprocess with a branch-marker assertion (§8). Every incumbent is re-checked against
the model's rows, box and integrality. Exits non-zero when no oracle check ran (§6).
Results: docs/dev/issue-1678-implied-bounds-panel-2026-10-10.md.
"""

import collections
import json
import os
import statistics as st
import subprocess
import sys

import discopt.modeling as dm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from panel_1654_coef_tighten import FAMILIES as FAMILIES_1654  # noqa: E402


def facility_open_eq(seed, M):
    rng = np.random.default_rng(seed)
    nf, nc = 3 + seed % 2, 4 + seed % 3
    f = rng.uniform(50, 150, nf)
    d = rng.uniform(10, 40, nc).round(2)
    c = rng.uniform(1, 6, (nf, nc))
    m = dm.Model(f"fleq{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{j}", lb=0) for j in range(nc)] for i in range(nf)]
    for j in range(nc):
        m.subject_to(sum(x[i][j] for i in range(nf)) == float(d[j]))
    for i in range(nf):
        for j in range(nc):
            m.subject_to(x[i][j] <= M * y[i])
    m.minimize(
        sum(float(f[i]) * y[i] for i in range(nf))
        + sum(float(c[i, j]) * x[i][j] for i in range(nf) for j in range(nc))
    )
    return m


facility_open_eq.M_ref = 40.0


def facility_open_cap(seed, M):
    rng = np.random.default_rng(1000 + seed)
    nf, nc = 4, 6
    cost = rng.uniform(1, 10, (nf, nc))
    fc = rng.uniform(5, 30, nf)
    dem = rng.uniform(1, 5, nc)
    cap = rng.uniform(8, 15, nf)
    m = dm.Model(f"flcap{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{c}", lb=0) for c in range(nc)] for i in range(nf)]
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


facility_open_cap.M_ref = 15.0


def facility_open_ge(seed, M):
    """No capacity and ``>=`` demand: ``x`` has no implied upper bound."""
    rng = np.random.default_rng(2000 + seed)
    nf, nc = 3, 5
    cost = rng.uniform(1, 10, (nf, nc))
    fc = rng.uniform(5, 30, nf)
    dem = rng.uniform(1, 5, nc)
    m = dm.Model(f"flge{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{c}", lb=0) for c in range(nc)] for i in range(nf)]
    for c in range(nc):
        m.subject_to(sum(x[i][c] for i in range(nf)) >= float(dem[c]))
    for i in range(nf):
        for c in range(nc):
            m.subject_to(x[i][c] <= M * y[i])
    m.minimize(
        sum(float(cost[i, c]) * x[i][c] for i in range(nf) for c in range(nc))
        + sum(float(fc[i]) * y[i] for i in range(nf))
    )
    return m


facility_open_ge.M_ref = 5.0


def fixed_charge_open(seed, M):
    rng = np.random.default_rng(3000 + seed)
    n = 8
    f = rng.uniform(10, 100, n)
    v = rng.uniform(1, 30, n)
    D = float(rng.uniform(20, 60))
    m = dm.Model(f"fco{seed}")
    x = [m.continuous(f"x{i}", lb=0) for i in range(n)]
    y = [m.binary(f"y{i}") for i in range(n)]
    m.subject_to(sum(x) >= D)
    for i in range(n):
        m.subject_to(x[i] <= M * y[i])
    m.minimize(sum(float(f[i]) * y[i] + float(v[i]) * x[i] for i in range(n)))
    return m


fixed_charge_open.M_ref = 60.0


def sequencing_open(seed, M):
    rng = np.random.default_rng(4000 + seed)
    n = 5
    p = rng.integers(2, 10, n)
    r = rng.integers(0, 9, n)
    m = dm.Model(f"seqo{seed}")
    s = [m.continuous(f"s{i}", lb=float(r[i])) for i in range(n)]
    C = m.continuous("C", lb=0)
    for i in range(n):
        for j in range(i + 1, n):
            y = m.binary(f"y{i}_{j}")
            m.subject_to(s[i] + float(p[i]) <= s[j] + M * (1 - y))
            m.subject_to(s[j] + float(p[j]) <= s[i] + M * y)
    for i in range(n):
        m.subject_to(s[i] + float(p[i]) <= C)
    m.minimize(C)
    return m


sequencing_open.M_ref = 100.0

FAMILIES = {
    **FAMILIES_1654,
    "facility_open_eq": facility_open_eq,
    "facility_open_cap": facility_open_cap,
    "facility_open_ge": facility_open_ge,
    "fixed_charge_open": fixed_charge_open,
    "sequencing_open": sequencing_open,
}

CHILD_FAMILY = r"""
import json, sys, time, warnings, contextlib, io
warnings.simplefilter("ignore")
sys.path.insert(0, HERE)
import discopt, discopt.solvers.lp_milp_highs as L
assert hasattr(L, "_implied_bounds_enabled"), "branch marker missing"
assert discopt.__file__.startswith(ROOT), discopt.__file__
from panel_1678_implied_bounds import FAMILIES
fam, seed, M = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
m = FAMILIES[fam](seed, M)
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    r = m.solve(time_limit=60)
wall = time.perf_counter() - t
viol = None
if r.x is not None:
    import numpy as np
    from discopt.solver import _make_evaluator, _infer_constraint_bounds
    ev = _make_evaluator(m); cl, cu = _infer_constraint_bounds(m, ev)
    x = np.concatenate([np.atleast_1d(np.asarray(r.x[v.name], float)).ravel() for v in m._variables])
    g = np.asarray(ev.evaluate_constraints(x), float)
    viol = float(np.max(np.maximum(np.maximum(np.asarray(cl) - g, g - np.asarray(cu)), 0))) if len(cl) else 0.0
    lb = np.concatenate([np.atleast_1d(np.asarray(v.lb, float)).ravel() for v in m._variables])
    ub = np.concatenate([np.atleast_1d(np.asarray(v.ub, float)).ravel() for v in m._variables])
    viol = max(viol, float(np.max(np.maximum(np.maximum(lb - x, x - ub), 0))))
    ints = [x[i] for i, v in enumerate([vv for vv in m._variables for _ in range(vv.size)]) if v.var_type.name in ("BINARY", "INTEGER")]
    viol = max(viol, max([abs(a - round(a)) for a in ints] or [0.0]))
st = r.solver_stats or {}
print(json.dumps(dict(status=r.status, obj=r.objective, bound=r.bound, cert=bool(r.gap_certified),
    wall=wall, viol=viol, nodes=int(r.node_count or 0),
    ct_entries=st.get("milp/coef_tightened_entries"), ct_ran=st.get("milp/coef_tighten_ran"))))
"""

CHILD_CHECK = r"""
import json, os, sys, tempfile, time, warnings, contextlib, io
warnings.simplefilter("ignore")
sys.path.insert(0, HERE)
import numpy as np
import discopt, discopt.solvers.lp_milp_highs as L
assert hasattr(L, "_implied_bounds_enabled"), "branch marker missing"
assert discopt.__file__.startswith(ROOT), discopt.__file__
import lp_milp_loader as LD
from discopt.modeling.core import from_nl
path = sys.argv[1]
inst = LD.read_instance(path)
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, inst.name + ".nl")
    perm = LD.write_nl(inst, nl)
    mdl = from_nl(nl)
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    r = mdl.solve(time_limit=60)
wall = time.perf_counter() - t
viol = None
if r.x is not None:
    x = np.empty(inst.n)
    for k in range(inst.n):
        x[perm[k]] = float(np.asarray(r.x[f"x{k}"], float).ravel()[0])
    viol = LD.dense_max_violation(inst, x)
st = r.solver_stats or {}
print(json.dumps(dict(status=r.status, obj=r.objective, bound=r.bound, cert=bool(r.gap_certified),
    wall=wall, viol=viol, nodes=int(r.node_count or 0), maximize=bool(inst.maximize),
    ct_entries=st.get("milp/coef_tightened_entries"), ct_ran=st.get("milp/coef_tighten_ran"))))
"""

ROOT = os.path.dirname(os.path.dirname(HERE))


def _child(src):
    return src.replace("HERE", repr(HERE)).replace("ROOT", repr(os.path.join(ROOT, "python")))


def _run(args, src, flag):
    env = dict(os.environ, DISCOPT_MILP_IMPLIED_BOUNDS=flag, PYTHONUNBUFFERED="1")
    p = subprocess.run(
        [sys.executable, "-c", _child(src), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not line:
        return dict(status="CRASH", err=p.stderr[-800:])
    return json.loads(line[-1])


def analyze(path):
    rows = [json.loads(ln) for ln in open(path)]
    bad = []
    stats = collections.defaultdict(collections.Counter)
    walls = collections.defaultdict(list)
    checks = 0
    for r in rows:
        key = (r["arm"], r["flag"])
        stats[key]["n"] += 1
        stats[key][r["status"]] += 1
        if r.get("cert"):
            stats[key]["certified"] += 1
        walls[key].append(r.get("wall", 0.0))
        if r["status"] == "CRASH":
            bad.append(("crash", r))
            continue
        ref = r.get("ref")
        if ref is None:
            continue
        checks += 1
        sgn = -1.0 if r.get("maximize") else 1.0  # internal minimize sense
        tol = 1e-6 * max(1.0, abs(ref))
        if r.get("bound") is not None and sgn * r["bound"] > sgn * ref + tol:
            bad.append(("bound crosses the reference", r))
        if r.get("obj") is not None:
            if r.get("viol") is None or r["viol"] > 1e-5:
                bad.append(("incumbent infeasible", r))
            elif sgn * r["obj"] < sgn * ref - tol:
                bad.append(("objective better than the reference", r))
        if r.get("cert") and r["status"] == "optimal":
            if r.get("obj") is None or abs(r["obj"] - ref) > 1e-4 * max(1.0, abs(ref)):
                bad.append(("certified wrong value", r))
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["arm"], r["inst"], r.get("M"))][r["flag"]] = r
    reg = [
        k for k, d in by.items() if d.get("0", {}).get("cert") and not d.get("1", {}).get("cert")
    ]
    gain = [
        k for k, d in by.items() if d.get("1", {}).get("cert") and not d.get("0", {}).get("cert")
    ]
    drift = [
        k
        for k, d in by.items()
        if d.get("0", {}).get("cert")
        and d.get("1", {}).get("cert")
        and abs(d["0"]["obj"] - d["1"]["obj"]) > 1e-6 * max(1.0, abs(d["0"]["obj"]))
    ]
    changed = [
        k for k, d in by.items() if d.get("0", {}).get("status") != d.get("1", {}).get("status")
    ]
    for key in sorted(stats):
        w = walls[key]
        print(
            f"arm={key[0]} flag={key[1]}: {dict(stats[key])}  "
            f"wall total {sum(w):.1f}s median {st.median(w):.2f}s"
        )
    print("certification regressions (ON loses a cert):", reg)
    print("certification gains:", len(gain), sorted({(k[0], k[1].split("/")[0]) for k in gain}))
    print("objective drift between certified arms:", drift)
    print("status changed:", len(changed))
    print("soundness violations:", len(bad))
    for why, r in bad[:30]:
        print(
            "  ",
            why,
            {
                k: r.get(k)
                for k in ("arm", "inst", "M", "flag", "status", "obj", "bound", "ref", "viol")
            },
        )
    print("executed oracle checks:", checks)
    return checks, bad, reg


def main():
    out = sys.argv[1]
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(4)
    ref_root = sys.argv[3] if len(sys.argv) > 3 else ROOT
    Ms = [1e4, 1e6, 1e8, 1e10, 1e14]
    n_done = 0
    with open(out, "w") as fh:

        def emit(rec):
            nonlocal n_done
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            print(
                n_done, rec["arm"], rec["inst"], rec.get("M"), rec["flag"], rec.get("status"),
                rec.get("obj"), rec.get("bound"), rec.get("cert"), rec.get("ct_entries"),
                f"{rec.get('wall', 0):.1f}s", flush=True,
            )  # fmt: skip

        for fam, b in FAMILIES.items():
            arm = "fam1654" if fam in FAMILIES_1654 else "open"
            for seed in seeds:
                ref = _run([fam, str(seed), repr(b.M_ref)], CHILD_FAMILY, "0")
                ref_obj = ref.get("obj") if ref.get("cert") else None
                for M in Ms:
                    order = ("0", "1") if (seed + Ms.index(M)) % 2 == 0 else ("1", "0")
                    for flag in order:
                        res = _run([fam, str(seed), repr(M)], CHILD_FAMILY, flag)
                        emit(dict(arm=arm, inst=f"{fam}/{seed}", M=M, flag=flag, ref=ref_obj,
                                  ref_status=ref.get("status"), **res))  # fmt: skip
        manifest = json.load(
            open(os.path.join(ROOT, "discopt_benchmarks/data/lp_milp/manifest.json"))
        )
        for k, mem in enumerate(manifest["members"]):
            p = mem["path"]
            p = os.path.join(ROOT if p.startswith("discopt_benchmarks") else ref_root, p)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            for flag in ("0", "1") if k % 2 == 0 else ("1", "0"):
                res = _run([p], CHILD_CHECK, flag)
                emit(dict(arm="check", inst=mem["name"], M=None, flag=flag,
                          ref=mem.get("oracle_objective"), **res))  # fmt: skip
    print("executed solves:", n_done)
    checks, bad, reg = analyze(out)
    sys.exit(0 if checks else 1)


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        main()
