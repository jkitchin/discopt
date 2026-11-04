"""#2136 G3.2 differential replay: captured Rust-MILP inputs, flag OFF vs ON
(interleaved per instance), judged by ``oracle.judge`` against the best independently
verified point across both arms and HiGHS (no solver's status is trusted).

usage: replay.py --flag {coef,tabcmir} [--profile {captured,strong}] [--base ENV=V ...]
                 T out.jsonl pkl...

``--flag coef``    toggles DISCOPT_RUST_MILP_COEF_TIGHTEN (step 1).
``--flag tabcmir`` toggles DISCOPT_RUST_MILP_TABLEAU_CMIR (step 2). Coefficient tightening
                   is its measured precondition, so pass ``--base
                   DISCOPT_RUST_MILP_COEF_TIGHTEN=1`` to hold it ON in both arms.
``--profile strong`` is the root profile of the gate doc: 200 cuts, 10 rounds, selection,
                   #2107's aggregated c-MIR on. ``captured`` keeps the captured options.

Each arm asserts its flag's firing counter (zero OFF, nonzero ON), so a row cannot pass
on an arm that never ran the mechanism. Exits non-zero when no row was judged."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

os.environ["DISCOPT_PROFILE"] = "1"
W = os.environ.get("DISCOPT_ROOT") or str(pathlib.Path(__file__).resolve().parents[3])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import discopt  # noqa: E402

assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R  # noqa: E402
import numpy as np  # noqa: E402
import oracle  # noqa: E402

_so = open(R.__file__, "rb").read()
for marker in (b"SepTabCmir", b"MilpCoefTightenRuns", b"MilpIncumbentMisvaluedDecert"):
    assert marker in _so, f"marker {marker!r} absent from {R.__file__}"

FLAGS = {
    "coef": ("DISCOPT_RUST_MILP_COEF_TIGHTEN", "MilpCoefTightenRuns"),
    "tabcmir": ("DISCOPT_RUST_MILP_TABLEAU_CMIR", "SepCmirCalls"),
}
STRONG = dict(root_cuts=200, cut_rounds=10, cut_select=True, cmir_cuts=True)
CTRS = (
    "MilpCoefTightenRuns",
    "MilpCoefTightenCoefs",
    "SepCmirCalls",
    "SepCmirCutsEmitted",
    "MilpIncumbentMisvaluedDecert",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", choices=sorted(FLAGS), required=True)
    ap.add_argument("--profile", choices=("captured", "strong"), default="captured")
    ap.add_argument("--base", action="append", default=[], help="ENV=V held in both arms")
    ap.add_argument("T", type=float)
    ap.add_argument("out")
    ap.add_argument("pkls", nargs="+")
    a = ap.parse_args()
    env, fire_ctr = FLAGS[a.flag]
    for kv in a.base:
        k, v = kv.split("=", 1)
        assert k != env, "the flag under test cannot be held in --base"
        os.environ[k] = v
    out = open(a.out, "a")
    nrows = nchecks = nbad = 0
    for p in a.pkls:
        d = pickle.load(open(p, "rb"))
        d.pop("had_debug_hook", None)
        name = os.path.basename(p)[:-4]
        t0 = time.perf_counter()
        h = oracle.highs_ref(d)
        ht = time.perf_counter() - t0
        hx = h.pop("x")
        h["viol"] = oracle.verify(d, hx) if hx is not None else None
        if hx is not None:
            h["obj"] = oracle.objective(d, hx)
        row = {
            "inst": name,
            "flag": env,
            "profile": a.profile,
            "base": a.base,
            "m": d["m"],
            "ns": d["n_struct"],
            "ints": len(d["integer_cols"]),
            "highs": {**h, "wall": round(ht, 2)},
            "load1": os.getloadavg()[0],
        }
        kw = dict(d)
        kw["time_limit_s"] = a.T
        kw["max_nodes"] = 10**9  # full run, not the captured probe budget
        if a.profile == "strong":
            kw.update(STRONG)
        if kw.get("initial_incumbent") is not None:
            kw["initial_incumbent"] = np.ascontiguousarray(kw["initial_incumbent"])
        arms = {}
        for arm in ("0", "1"):
            os.environ[env] = arm
            rk = dict(kw, max_nodes=2, time_limit_s=min(a.T, 60.0))
            root = R.solve_milp_csc_py(**rk)[3]
            R.profile_reset_py()
            t0 = time.perf_counter()
            st, x, obj, bd, nodes, its = R.solve_milp_csc_py(**kw)
            w = time.perf_counter() - t0
            pc = R.profile_counters_py()
            ctr = {k: pc.get(k, 0) for k in CTRS}
            if arm == "0":
                assert ctr[fire_ctr] == 0, (name, arm, ctr)
            else:
                assert ctr[fire_ctr] >= 1, (name, arm, ctr)
            v = oracle.verify(d, x) if np.isfinite(obj) and len(x) else None
            arms[f"arm{arm}"] = {
                "status": st,
                "obj": obj,
                "bound": bd,
                "root": root,
                "nodes": nodes,
                "iters": its,
                "wall": round(w, 2),
                "ctr": ctr,
                "viol": v,
            }
        os.environ.pop(env, None)
        verdict = oracle.judge(h, arms, gap_tol=d["gap_tol"], abs_gap_tol=d.get("abs_gap_tol"))
        row.update(arms)
        row.update(verdict)
        out.write(json.dumps(row, default=float) + "\n")
        out.flush()
        nrows += 1
        nchecks += verdict["checks"]
        nbad += bool(verdict["bad"])
        a0, a1 = arms["arm0"], arms["arm1"]
        print(
            f"{name:42s} ref {verdict['best_primal']:12.6g} ({verdict['best_from']}) | "
            f"root {a0['root']:12.6g} -> {a1['root']:12.6g} | "
            f"OFF {a0['status'][:9]:9s} b {a0['bound']:12.6g} n {a0['nodes']:7d} {a0['wall']:6.2f}s | "
            f"ON {a1['status'][:9]:9s} b {a1['bound']:12.6g} n {a1['nodes']:7d} {a1['wall']:6.2f}s"
            + (f" BAD {verdict['bad']}" if verdict["bad"] else "")
            + (f" NOTE {verdict['notes']}" if verdict["notes"] else ""),
            flush=True,
        )
    print(f"PROBE_DONE rows={nrows} checks={nchecks} bad_rows={nbad}")
    return 0 if nrows and nchecks else 1


if __name__ == "__main__":
    sys.exit(main())
