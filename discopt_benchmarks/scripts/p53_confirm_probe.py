"""highs-parity P5.3: which second engine catches a false master certificate?

On every captured OA master (``python/tests/data/oa_masters/*.npz`` and the #2069
``clay0203_master.mps.gz``) each engine configuration solves the master once:

``R``      the Rust simplex B&B as the decomposition/route primary calls it
           (``milp_simplex.solve_milp``, presolve on);
``R_alt``  the same driver with presolve OFF and heuristics OFF -- the "Rust re-solve,
           different configuration" confirmer the plan names (the driver has no seed
           knob, so presolve and heuristics are what differ);
``H_on`` / ``H_off``  one HiGHS solve with presolve on / off (``milp_highs._solve_milp_once``).

Ground truth (``TRUTH``) is the recorded independent optimum where one exists (SCIP,
#2069/#2115); otherwise the optimum both engine families agree on is recorded as
``agreed`` and marked as such. A *false certificate* is a claim of ``optimal`` whose
bound exceeds the truth by more than the gap tolerance. A confirmer *catches* a false
primary certificate when its own valid bound is below the primary's claim by more than
the gap (the shipped chain then withdraws or lowers it). Prints executed count; exits
non-zero on 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import master_parity_panel as MP  # noqa: E402, N812

MASTERS = [
    REPO / "python/tests/data/issue2069/clay0203_master.mps.gz",
    *sorted((REPO / "python/tests/data/oa_masters").glob("*.npz")),
]
#: Independent optima (SCIP), from the issue records cited in milp_highs.py.
TRUTH = {"clay0203_master": 41420.27, "m6_final_master_2115": 82.26}
GAP = 1e-4


def run(cfg: str, kw: dict, tl: float) -> dict:
    t0 = time.perf_counter()
    if cfg == "R":
        from discopt.solvers.milp_simplex import solve_milp

        r = solve_milp(**{**kw, "time_limit": tl, "gap_tolerance": GAP})
        out = (str(r.status), r.objective, getattr(r, "bound", r.objective))
    elif cfg == "R_alt":
        from discopt._rust import solve_milp_csc_py
        from discopt.solvers.milp_simplex import _marshal_std_form

        s = _marshal_std_form(kw["c"], kw.get("A_ub"), kw.get("b_ub"), kw.get("A_eq"),
                              kw.get("b_eq"), kw.get("bounds"), kw.get("integrality"))  # fmt: skip
        n = np.asarray(kw["c"]).ravel().shape[0]
        st, _x, obj, bnd, _nodes, _it = solve_milp_csc_py(
            s.c, s.m, n + s.m, s.col_ptr, s.row_idx, s.vals, s.b, s.lb, s.ub, s.int_cols, n,
            0.0, 10_000_000, GAP, presolve=False, heuristics=False, time_limit_s=tl,
        )  # fmt: skip
        out = (st, obj, bnd)
    else:
        import discopt.solvers.milp_highs as mh

        kw2 = {k: kw.get(k) for k in ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds",
                                      "integrality")}  # fmt: skip
        r = mh._solve_milp_once(**kw2, time_limit=tl, gap_tolerance=GAP, mip_start=None,
                               presolve=(cfg == "H_on"))  # fmt: skip
        out = (str(r.status), r.objective, r.bound)
    return {"status": out[0], "obj": out[1], "bound": out[2], "wall": time.perf_counter() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, executed = [], 0
    for p in MASTERS:
        name = p.name.split(".")[0]
        kw = MP.fixture_kw(p)
        kw.pop("mip_start", None)
        row = {"master": name}
        for cfg in ("R", "R_alt", "H_on", "H_off"):
            row[cfg] = run(cfg, kw, a.time_limit)
            executed += 1
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "truth": TRUTH}, indent=1, default=float))
    print(f"p53: executed={executed}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
