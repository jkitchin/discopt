"""Issue #694 entry experiment — bound-vs-build-time curve for the root McCormick
relaxation build.

Measurement-only (no library change; mirrors ``pf_panel.py`` #632). The hypothesis
(#694) is that the monolithic ``build_uniform_relaxation`` produces its dual bound
only at the *end* of the build, so truncating the build (to honor ``time_limit``)
loses the bound (baron-gap-plan.md §8.1). If instead a *finite* LP bound exists
early in the build, an interrupted build still yields a valid (weaker) bound and the
§8 fork dissolves.

What this probe does, per instance:

1. Instrument ``_Builder.add_row`` to timestamp every row as it is appended during a
   single full ``build_uniform_relaxation`` — this gives ``elapsed(row_i)``, the
   build wall-clock at which row ``i`` first existed, with **zero** checkpoint-solve
   perturbation (the build runs once, uninterrupted).
2. Post-hoc, at each build decile ``k`` (10%, 20%, …, 100% of the final rows), solve
   the *prefix* LP ``A_ub[:n_k] x <= b[:n_k]`` over the full column set and the real
   objective, and record ``(elapsed(n_k), n_k, status, bound, finite?)``. A prefix
   relaxation has FEWER rows, so it is still a valid outer approximation — its LP min
   is a valid (weaker) lower bound whenever it is finite (bounded). This is exactly
   the "weaken but never falsify" property the hypothesis rests on.
3. Also record the always-available sound interval floor (``milp._objective_floor``,
   the ``obj_box_lb`` computed from cost-column bounds alone), which needs zero rows.

Kill criterion (#694): the approach dies if, on the #654 class, the partial-build
bound is ``-inf``/``None`` until ≳90% of build time — no useful anytime curve.

Usage::

    python discopt_benchmarks/scripts/issue694_anytime_build_probe.py \
        casctanks hda heatexch_gen1 nvs05 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
    "tests",
    "data",
    "minlplib_nl",
)
_BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


def _resolve(inst: str) -> str | None:
    for root in (_CORPUS, _BENCH):
        p = os.path.join(root, inst + ".nl")
        if os.path.exists(p):
            return p
    return None


def _root_box(model):
    lb = np.array([v.lb if v.lb is not None else -1e20 for v in model._variables], float)
    ub = np.array([v.ub if v.ub is not None else 1e20 for v in model._variables], float)
    return lb, ub


def probe_instance(inst: str, solve_tl: float = 20.0) -> dict:
    from discopt.modeling.core import from_nl

    path = _resolve(inst)
    if path is None:
        return {"instance": inst, "status": "missing"}

    model = from_nl(path)
    lb, ub = _root_box(model)
    return probe_model(model, inst, lb, ub, solve_tl=solve_tl)


def probe_model(model, inst, lb, ub, solve_tl: float = 20.0) -> dict:
    from discopt._jax import uniform_relax as ur
    from discopt._jax.milp_relaxation import MilpRelaxationModel

    # --- 1. instrument add_row with timestamps, run ONE uninterrupted build -----
    row_stamps: list[float] = []
    orig_add_row = ur._Builder.add_row
    build_t0 = [0.0]

    def timed_add_row(self, coeffs, rhs):
        orig_add_row(self, coeffs, rhs)
        # record only when a row is actually appended (add_row drops all-zero rows)
        if len(row_stamps) < len(self.rows):
            row_stamps.append(time.perf_counter() - build_t0[0])

    ur._Builder.add_row = timed_add_row
    try:
        build_t0[0] = time.perf_counter()
        rel = ur.build_uniform_relaxation(model, (lb, ub))
        build_wall = time.perf_counter() - build_t0[0]
    finally:
        ur._Builder.add_row = orig_add_row

    milp = rel.model
    c = np.asarray(milp._c, dtype=np.float64)
    obj_offset = float(milp._obj_offset) if hasattr(milp, "_obj_offset") else 0.0
    # MilpRelaxationModel stores obj_offset; read via the attribute the class uses.
    obj_offset = float(getattr(milp, "_obj_offset", getattr(milp, "obj_offset", 0.0)))
    bounds = list(milp._bounds) if getattr(milp, "_bounds", None) is not None else None
    A = milp._A_ub
    b = milp._b_ub
    n_rows = 0 if A is None else int(A.shape[0])
    obj_floor = milp._objective_floor
    obj_valid = bool(milp._objective_bound_valid)

    # Align stamp count with the assembled matrix (add_row may append the separable
    # floor row after the constraint loop; both are part of the build).
    while len(row_stamps) < n_rows:
        row_stamps.append(build_wall)
    row_stamps = row_stamps[:n_rows]

    import scipy.sparse as sp

    Acsr = None if A is None else sp.csr_matrix(A)

    def solve_prefix(nk: int) -> dict:
        if Acsr is None or nk <= 0:
            Ap, bp = None, None
        else:
            Ap = Acsr[:nk]
            bp = b[:nk]
        m = MilpRelaxationModel(
            c=c,
            A_ub=Ap,
            b_ub=bp,
            bounds=bounds,
            obj_offset=obj_offset,
            integrality=None,
            objective_bound_valid=obj_valid,
        )
        t0 = time.perf_counter()
        res = m.solve(backend="simplex", time_limit=solve_tl, gap_tolerance=1e-6)
        solve_wall = time.perf_counter() - t0
        bound = res.bound
        finite = (
            res.status == "optimal" and bound is not None and math.isfinite(float(bound))
        )
        return {
            "n_rows": nk,
            "frac_rows": (nk / n_rows) if n_rows else 1.0,
            "build_elapsed": row_stamps[nk - 1] if nk > 0 else 0.0,
            "build_frac": (
                (row_stamps[nk - 1] / build_wall) if (nk > 0 and build_wall > 0) else 0.0
            ),
            "status": res.status,
            "bound": None if bound is None else float(bound),
            "finite": bool(finite),
            "solve_wall": solve_wall,
        }

    # --- 2. checkpoints at every 10% of rows -----------------------------------
    checkpoints = []
    for k in range(1, 11):
        nk = int(math.ceil(k / 10.0 * n_rows))
        nk = max(1, min(nk, n_rows))
        checkpoints.append(solve_prefix(nk))

    # first build-decile at which a finite bound appears (build_frac of that row)
    first_finite = next((cp for cp in checkpoints if cp["finite"]), None)

    return {
        "instance": inst,
        "status": "ok",
        "n_vars": len(model._variables),
        "n_cons": len(model._constraints),
        "n_rows": n_rows,
        "n_cols": int(c.size),
        "build_wall": build_wall,
        "obj_floor": None if obj_floor is None else float(obj_floor),
        "obj_bound_valid": obj_valid,
        "full_bound": checkpoints[-1]["bound"] if checkpoints else None,
        "first_finite_build_frac": None if first_finite is None else first_finite["build_frac"],
        "first_finite_row_frac": None if first_finite is None else first_finite["frac_rows"],
        "checkpoints": checkpoints,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--solve-tl", type=float, default=20.0)
    args = ap.parse_args()

    results = []
    for inst in args.instances:
        print(f"\n=== {inst} ===", flush=True)
        r = probe_instance(inst, solve_tl=args.solve_tl)
        results.append(r)
        if r["status"] != "ok":
            print(f"  {r['status']}", flush=True)
            continue
        print(
            f"  vars={r['n_vars']} cons={r['n_cons']} rows={r['n_rows']} cols={r['n_cols']} "
            f"build_wall={r['build_wall']:.3f}s obj_floor={r['obj_floor']} "
            f"full_bound={r['full_bound']}",
            flush=True,
        )
        print("  build%  row%   elapsed   status     bound         finite", flush=True)
        for cp in r["checkpoints"]:
            bstr = "None" if cp["bound"] is None else f"{cp['bound']:.6g}"
            print(
                f"  {cp['build_frac']*100:5.1f}  {cp['frac_rows']*100:5.1f}  "
                f"{cp['build_elapsed']:7.3f}  {cp['status']:>10}  "
                f"{bstr:>12}  {cp['finite']}",
                flush=True,
            )
        ff = r["first_finite_build_frac"]
        print(
            f"  --> first finite bound at build_frac="
            f"{'never' if ff is None else f'{ff*100:.1f}%'}",
            flush=True,
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
