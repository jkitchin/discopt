"""#1733 graduation panel: DISCOPT_MILP_ROW_FORM ON vs OFF.

Usage:
    python -u discopt_benchmarks/scripts/panel_1733_row_form.py OUT.jsonl [n_seeds] [ref_root]
    python discopt_benchmarks/scripts/panel_1733_row_form.py analyze OUT.jsonl

Three arms, each with the flag OFF and ON, interleaved, with the arm order alternating:

* ``eps``: the issue's class -- an epsilon-constraint facility location MILP (5 sites,
  7 customers, all binary): minimise cost subject to ``emissions <= e* + shift`` where
  ``e*`` is the minimum emission, for ``shift`` in {1e-6, 0, 1e-4}, solved cold and
  warm-started from the emission-optimal point (``discopt.mo.epsilon_constraint``'s
  default). The oracle is exhaustive enumeration of all 5**7 assignments in integer
  arithmetic (every emission coefficient is a multiple of 0.2), so it does not depend
  on any solver;
* the #1654/#1678 big-M families and the LP/MILP check instances of
  ``discopt_benchmarks/data/lp_milp/manifest.json``, re-used from
  ``panel_1678_implied_bounds.py`` with the same oracles (the flag-OFF solve at the
  family's ``M_ref``; the manifest's oracle objective).

Every solve runs in a fresh subprocess with a branch-marker assertion (CLAUDE.md §8);
every incumbent is re-checked against the model's rows, box and integrality. Exits
non-zero when no oracle check ran (§6).
Results: docs/dev/issue-1733-row-form-panel-2026-10-17.md.
"""

import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import panel_1678_implied_bounds as p1678  # noqa: E402

N_SITES, N_CUST = 5, 7
SHIFTS = (1e-6, 0.0, 1e-4)


def eps_data(seed):
    rng = np.random.default_rng(seed)
    sites = rng.uniform(0, 10, (N_SITES, 2))
    cust = rng.uniform(0, 10, (N_CUST, 2))
    dist = np.linalg.norm(sites[:, None] - cust[None], axis=2)
    d = rng.integers(5, 15, N_CUST).astype(float)
    phi = np.array([30.0, 40.0, 55.0, 70.0, 90.0])
    e = np.array([3.0, 2.4, 1.6, 1.0, 0.4])
    u = np.full(N_SITES, 35.0)
    return dist, d, phi, e, u


def eps_oracle(seed):
    """``(e*, cost optimum)`` by enumerating every customer-to-site assignment; a site
    is opened iff it serves someone, which is optimal since ``phi > 0``."""
    dist, d, phi, e, u = eps_data(seed)
    assign = np.array(list(itertools.product(range(N_SITES), repeat=N_CUST)))
    di = d.astype(int)
    e5 = np.round(e * 5).astype(int)
    assert np.allclose(e5, e * 5)
    load = np.zeros((len(assign), N_SITES), int)
    for i in range(N_CUST):
        np.add.at(load, (np.arange(len(assign)), assign[:, i]), di[i])
    ok = (load <= u.astype(int)).all(1)
    em5 = sum(e5[assign[:, i]] * di[i] for i in range(N_CUST))
    emin = int(em5[ok].min())
    feas = ok & (em5 == emin)
    cost = (phi[None] * (load > 0)).sum(1) + sum(
        0.5 * dist[assign[:, i], i] * d[i] for i in range(N_CUST)
    )
    return emin / 5.0, float(cost[feas].min())


CHILD_EPS = r"""
import json, sys, time, warnings, contextlib, io
warnings.simplefilter("ignore")
sys.path.insert(0, HERE)
import numpy as np
import discopt, discopt.solvers.lp_milp_highs as L
assert hasattr(L, "_eliminate_logicals"), "branch marker missing"
assert discopt.__file__.startswith(ROOT), discopt.__file__
from panel_1733_row_form import eps_data, N_SITES, N_CUST
seed, shift, mode = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
dist, d, phi, e, u = eps_data(seed)
m = discopt.Model("eps")
y = m.binary("y", shape=(N_SITES,)); x = m.binary("x", shape=(N_SITES, N_CUST))
for i in range(N_CUST):
    m.subject_to(sum(x[j, i] for j in range(N_SITES)) == 1)
for j in range(N_SITES):
    m.subject_to(sum(d[i] * x[j, i] for i in range(N_CUST)) <= u[j] * y[j])
em = sum(e[j] * d[i] * x[j, i] for j in range(N_SITES) for i in range(N_CUST))
with contextlib.redirect_stdout(io.StringIO()):
    m.minimize(em); r0 = m.solve()
m.subject_to(em <= r0.objective + shift)
m.minimize(sum(phi[j] * y[j] for j in range(N_SITES))
           + sum(0.5 * dist[j, i] * d[i] * x[j, i] for j in range(N_SITES) for i in range(N_CUST)))
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    if mode == "warm":
        r = m.solve(time_limit=60, initial_solution={y: r0.x["y"], x: r0.x["x"]})
    else:
        r = m.solve(time_limit=60)
wall = time.perf_counter() - t
viol = None
if r.x is not None:
    xx, yy = np.asarray(r.x["x"], float), np.asarray(r.x["y"], float)
    viol = max(float(np.max(np.abs(xx.sum(0) - 1))),
               float(np.max(np.maximum(d @ xx.T - u * yy, 0))),
               float(max(np.sum(e[:, None] * d[None] * xx) - (r0.objective + shift), 0)),
               float(np.max(np.abs(xx - np.round(xx)))), float(np.max(np.abs(yy - np.round(yy)))))
st = r.solver_stats or {}
print(json.dumps(dict(status=r.status, obj=r.objective, bound=r.bound, cert=bool(r.gap_certified),
    wall=wall, viol=viol, nodes=int(r.node_count or 0), r0=r0.objective,
    row_form=st.get("milp/logicals_as_row_bounds"))))
"""


def _run(args, src, flag):
    env = dict(os.environ, DISCOPT_MILP_ROW_FORM=flag, PYTHONUNBUFFERED="1")
    src = p1678._child(src).replace("_implied_bounds_enabled", "_eliminate_logicals")
    p = subprocess.run(
        [sys.executable, "-c", src, *args], env=env, capture_output=True, text=True, timeout=600
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not line:
        return {"status": "CRASH", "err": p.stderr[-800:]}
    return json.loads(line[-1])


def main():
    out = sys.argv[1]
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(4)
    ref_root = sys.argv[3] if len(sys.argv) > 3 else p1678.ROOT
    big_ms = [1e4, 1e6, 1e8, 1e10, 1e14]
    n_done = 0
    with open(out, "w") as fh:

        def emit(rec):
            nonlocal n_done
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            print(
                n_done, rec["arm"], rec["inst"], rec.get("M"), rec["flag"], rec.get("status"),
                rec.get("obj"), rec.get("bound"), rec.get("cert"), f"{rec.get('wall', 0):.1f}s",
                flush=True,
            )  # fmt: skip

        for seed in seeds:
            emin, opt = eps_oracle(seed)
            for k, (shift, mode) in enumerate(itertools.product(SHIFTS, ("cold", "warm"))):
                for flag in ("0", "1") if (seed + k) % 2 == 0 else ("1", "0"):
                    res = _run([str(seed), repr(shift), mode], CHILD_EPS, flag)
                    if res.get("r0") is not None:
                        assert abs(res["r0"] - emin) <= 1e-6, (seed, res["r0"], emin)
                    emit(dict(arm=f"eps-{mode}", inst=f"eps/{seed}", M=shift, flag=flag,
                              ref=opt, **res))  # fmt: skip
        for fam, b in p1678.FAMILIES.items():
            arm = "fam1654" if fam in p1678.FAMILIES_1654 else "open"
            for seed in seeds:
                ref = _run([fam, str(seed), repr(b.M_ref)], p1678.CHILD_FAMILY, "0")
                ref_obj = ref.get("obj") if ref.get("cert") else None
                for big_m in big_ms:
                    order = ("0", "1") if (seed + big_ms.index(big_m)) % 2 == 0 else ("1", "0")
                    for flag in order:
                        res = _run([fam, str(seed), repr(big_m)], p1678.CHILD_FAMILY, flag)
                        emit(dict(arm=arm, inst=f"{fam}/{seed}", M=big_m, flag=flag, ref=ref_obj,
                                  ref_status=ref.get("status"), **res))  # fmt: skip
        with open(os.path.join(p1678.ROOT, "discopt_benchmarks/data/lp_milp/manifest.json")) as mf:
            manifest = json.load(mf)
        for k, mem in enumerate(manifest["members"]):
            p = mem["path"]
            p = os.path.join(p1678.ROOT if p.startswith("discopt_benchmarks") else ref_root, p)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            for flag in ("0", "1") if k % 2 == 0 else ("1", "0"):
                res = _run([p], p1678.CHILD_CHECK, flag)
                emit(dict(arm="check", inst=mem["name"], M=None, flag=flag,
                          ref=mem.get("oracle_objective"), **res))  # fmt: skip
    print("executed solves:", n_done)
    checks, bad, reg = p1678.analyze(out)
    sys.exit(0 if checks else 1)


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        p1678.analyze(sys.argv[2])
    else:
        main()
