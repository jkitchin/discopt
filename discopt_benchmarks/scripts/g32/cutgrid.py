"""#2136 step 5: the Rust root cut-loop budget (total cuts x rounds x selection), GMI +
cover only, over every captured Rust-MILP input -- not just MTP.

For each input: the bare LP bound (``root_cuts=0``), HiGHS's point verified on the
original rows as the primal reference, and the root bound (``max_nodes=2``) for every
budget in GRID. Reports root gap closed = (root - lp) / (ref - lp). Every root is
checked against the verified reference (a root above it is a false bound and fails the
run). Coefficient tightening is held at ``$DISCOPT_RUST_MILP_COEF_TIGHTEN`` (default 1:
the gate doc measured it as the precondition); both c-MIR separators are OFF.

usage: cutgrid.py out.jsonl pkl...      (prints PROBE_DONE rows=N checks=M)"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("DISCOPT_RUST_MILP_COEF_TIGHTEN", "1")
os.environ["DISCOPT_RUST_MILP_TABLEAU_CMIR"] = "0"
W = os.environ.get("DISCOPT_ROOT") or str(pathlib.Path(__file__).resolve().parents[3])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import discopt  # noqa: E402

assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R  # noqa: E402
import numpy as np  # noqa: E402
import oracle  # noqa: E402

assert b"MilpIncumbentMisvaluedDecert" in open(R.__file__, "rb").read(), "marker absent"

# (total root cuts, rounds, select). 200/10/select is `_STRONG_CUT_PROFILE`, the
# escalation default; 3000/15 is the prototype's 200-per-round x 15-round volume.
GRID = [
    (16, 1, False),
    (200, 10, True),
    (1000, 10, True),
    (3000, 15, True),
    (3000, 15, False),
    (10**6, 30, True),
    (10**6, 30, False),
]
ROOT_T = float(os.environ.get("G32_ROOT_T", "120"))


def root(d, rc, rounds, sel):
    kw = dict(d)
    kw.update(
        root_cuts=rc,
        cut_rounds=rounds,
        cut_select=sel,
        gmi_cuts=True,
        cmir_cuts=False,
        max_nodes=2,
        time_limit_s=ROOT_T,
        root_cut_time_s=None,
        initial_incumbent=None,
    )
    t = time.perf_counter()
    r = R.solve_milp_csc_py(**kw)
    return r[3], time.perf_counter() - t


def main() -> int:
    out = open(sys.argv[1], "a")
    nrows = nchecks = nfalse = 0
    for p in sys.argv[2:]:
        d = pickle.load(open(p, "rb"))
        d.pop("had_debug_hook", None)
        name = os.path.basename(p)[:-4]
        h = oracle.highs_ref(d)
        hx = h.pop("x")
        if hx is None or oracle.verify(d, hx) > oracle.VIOL_TOL:
            print(f"{name:42s} SKIP no verified HiGHS point ({h['status']})", flush=True)
            continue
        ref = oracle.objective(d, hx)
        lp, _ = root(d, 0, 0, False)
        row = {"inst": name, "ref": ref, "highs_status": h["status"], "lp": lp, "arms": {}}
        cells = []
        for rc, rounds, sel in GRID:
            b, w = root(d, rc, rounds, sel)
            nchecks += 1
            false = b > ref + 1e-6 * max(1.0, abs(ref))
            nfalse += false
            denom = ref - lp
            closed = (b - lp) / denom if np.isfinite(lp) and np.isfinite(b) and denom > 1e-9 else None
            row["arms"][f"{rc}/{rounds}/{int(sel)}"] = {
                "root": b,
                "closed": closed,
                "wall": round(w, 3),
                "false": bool(false),
            }
            cells.append(
                ("FALSE " if false else "")
                + (f"{closed:6.3f}" if closed is not None else "   n/a")
                + f"({w:5.1f}s)"
            )
        out.write(json.dumps(row, default=float) + "\n")
        out.flush()
        nrows += 1
        print(f"{name:42s} lp {lp:12.6g} ref {ref:12.6g} | " + " ".join(cells), flush=True)
    print(f"PROBE_DONE rows={nrows} checks={nchecks} false_bounds={nfalse}")
    return 0 if nrows and nchecks and not nfalse else 1


if __name__ == "__main__":
    sys.exit(main())
