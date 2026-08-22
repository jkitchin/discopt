"""EXPERIMENT 2 -- how far does it scale, and does global search actually buy
anything over the local methods the field uses?

Part A: dimension scaling.  A real MECP problem has 3N-6 internal
coordinates: 3 for a triatomic, 12 for a 6-atom molecule, 30 for a 12-atom
molecule.  Solve the TwoWell spin-crossing MECP for n = 2..N_MAX and record
wall time, node count, certification status, and (for n <= 2) agreement with
the exact grid oracle / (for n > 2) agreement with a stochastic seam oracle.

Part B: global vs local.  Run the two standard local MECP algorithms from
random starting geometries on the same n=2 model that has two energetically
distinct seam basins:
   * Levine/Coe/Martinez penalty method (sigma escalation, BFGS)
   * direct constrained optimization (SLSQP on min W1 s.t. W1-W2=0)
and record how often each lands in the *global* basin.  This measures the
thing discopt would be replacing: the probability that a local MECP search
reports a crossing point that is not the lowest one.

Both parts print per-item progress and an executed-work counter.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import mecp_models as M  # noqa: E402

CHECKS = 0
NOTES: list[str] = []
OUT = {"scaling": [], "local": {}}

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TIME_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

# ==========================================================================
print("=" * 78)
print(f"PART A -- dimension scaling (time_limit={TIME_LIMIT}s per instance)")
print("=" * 78)
print(
    f"{'n':>3} {'status':<12} {'objective':>12} {'bound':>12} {'gap':>10} "
    f"{'cert':>5} {'nodes':>8} {'wall_s':>8} {'oracle':>12} {'d(obj-orc)':>11}"
)

for n in range(2, N_MAX + 1):
    tw = M.TwoWellParams(n=n)

    # independent oracle
    if n == 2:
        e_or, x_or, ncross, _ = M.seam_oracle_2d(tw, n_grid=1601)
        oracle_kind = "grid(exact)"
        assert ncross > 0
    else:
        e_or, x_or, npj = M.seam_oracle_sampled(tw, n_samples=250_000, n_refine=400, seed=7)
        oracle_kind = f"sampled({npj})"

    m, qs = M.build_spin_crossing_mecp(tw)
    t0 = time.time()
    r = m.solve(time_limit=TIME_LIMIT)
    wall = time.time() - t0

    d = None if r.objective is None else r.objective - e_or
    c_obj = f"{r.objective:12.6f}" if r.objective is not None else "        None"
    c_bnd = f"{r.bound:12.6f}" if r.bound is not None else "        None"
    c_gap = f"{r.gap:10.2e}" if r.gap is not None else "      None"
    c_d = f"{d:11.2e}" if d is not None else "       None"
    print(
        f"{n:>3} {r.status:<12} {c_obj} {c_bnd} {c_gap} "
        f"{str(r.gap_certified):>5} {r.node_count:>8} {wall:>8.2f} "
        f"{e_or:>12.6f} {c_d}",
        flush=True,
    )

    # A certified solve must not be WORSE than a stochastic upper bound,
    # and its bound must not exceed that upper bound either (soundness).
    CHECKS += 1
    if r.objective is not None:
        if r.objective > e_or + 1e-4:
            NOTES.append(
                f"n={n}: discopt objective {r.objective:.6f} worse than "
                f"{oracle_kind} oracle {e_or:.6f} -- missed a better seam point"
            )
        CHECKS += 1
    if r.bound is not None and r.gap_certified:
        CHECKS += 1
        if r.bound > e_or + 1e-4:
            NOTES.append(
                f"n={n}: SOUNDNESS -- certified bound {r.bound:.6f} exceeds a "
                f"known feasible seam point {e_or:.6f} from {oracle_kind}"
            )

    OUT["scaling"].append(
        {
            "n": n,
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "gap": r.gap,
            "gap_certified": bool(r.gap_certified),
            "nodes": int(r.node_count),
            "wall": wall,
            "oracle": e_or,
            "oracle_kind": oracle_kind,
        }
    )
    if wall > TIME_LIMIT * 0.95 and not r.gap_certified:
        print(f"    (n={n} hit the time limit uncertified; stopping the sweep)")
        break

# ==========================================================================
print("\n" + "=" * 78)
print("PART B -- local MECP algorithms from random starts (n=2, two basins)")
print("=" * 78)

tw2 = M.TwoWellParams(n=2)
e_glob, x_glob, _, basins = M.seam_oracle_2d(tw2, n_grid=1601)
e_local2 = basins[1][0] if len(basins) > 1 else None
print(f"global MECP  E={e_glob:.6f} at {np.round(x_glob, 4)}")
print(f"2nd basin    E={e_local2:.6f} at {np.round(basins[1][1], 4)}")
bnds = np.array(tw2.bounds())


def W(x):
    return tw2.states(list(np.asarray(x, float)), exp=np.exp)


def penalty_method(x0, sigmas=(0.5, 2.0, 8.0, 32.0, 128.0), alpha=0.02):
    """Levine/Coe/Martinez sequential smooth penalty, BFGS inner solves."""
    x = np.asarray(x0, float).copy()
    for sigma in sigmas:

        def F(z):
            w1, w2 = W(np.clip(z, bnds[:, 0], bnds[:, 1]))
            d = w2 - w1
            return 0.5 * (w1 + w2) + sigma * d * d / (abs(d) + alpha)

        res = minimize(F, x, method="L-BFGS-B", bounds=list(map(tuple, bnds)))
        x = res.x
    return x


def slsqp_constrained(x0):
    """Direct constrained form: min W1 s.t. W1-W2 = 0."""
    res = minimize(
        lambda z: W(z)[0],
        np.asarray(x0, float),
        method="SLSQP",
        bounds=list(map(tuple, bnds)),
        constraints=[{"type": "eq", "fun": lambda z: W(z)[0] - W(z)[1]}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x


rng = np.random.default_rng(12345)
N_START = 200
tally = {}
for name, fn in (("penalty(LCM)", penalty_method), ("SLSQP(constrained)", slsqp_constrained)):
    hits = converged = 0
    energies = []
    t0 = time.time()
    for k in range(N_START):
        x0 = rng.uniform(bnds[:, 0], bnds[:, 1])
        try:
            xf = fn(x0)
        except Exception as exc:
            NOTES.append(f"{name}: start {k} raised {type(exc).__name__}: {exc}")
            continue
        w1, w2 = W(xf)
        gap = abs(w1 - w2)
        CHECKS += 1
        if gap > 1e-3:  # never reached the seam
            continue
        converged += 1
        energies.append(float(w1))
        if abs(float(w1) - e_glob) < 1e-3:
            hits += 1
    el = time.time() - t0
    frac = hits / max(converged, 1)
    print(
        f"  {name:<20s} reached seam {converged:3d}/{N_START}  "
        f"found GLOBAL basin {hits:3d}/{converged} ({100 * frac:5.1f}%)  "
        f"[{el:.1f}s total]"
    )
    if energies:
        e = np.array(energies)
        print(
            f"      converged energies: min={e.min():.6f} median={np.median(e):.6f} "
            f"max={e.max():.6f}  distinct(1e-3)={len(np.unique(np.round(e, 3)))}"
        )
    tally[name] = {
        "starts": N_START,
        "reached_seam": converged,
        "found_global": hits,
        "frac_global": frac,
        "energies": energies,
    }

OUT["local"] = tally
OUT["global_energy"] = e_glob
OUT["second_basin_energy"] = e_local2

# ==========================================================================
print("\n" + "=" * 78)
print(f"EXECUTED CHECKS: {CHECKS}")
print(f"NOTES/ANOMALIES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
here = __file__.rsplit("/", 1)[0]
with open(f"{here}/exp2_results.json", "w") as fh:
    json.dump(OUT, fh, indent=2, default=float)
print(f"wrote {here}/exp2_results.json")
if CHECKS == 0:
    print("PROBE FIRED NOTHING")
    sys.exit(2)
