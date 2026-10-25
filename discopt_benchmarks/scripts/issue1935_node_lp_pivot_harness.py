"""#1935: replay captured node LPs through the in-house simplex, one variant per run.

The replay against HiGHS (``issue1935_highs_node_lp_replay.py``) left a pivot gap on
``pooling_bental5stp``. Comparing simplex variants in-tree is confounded: a
different pivot path lands on a different optimal vertex and the trees diverge.
This harness fixes the inputs instead:

``capture`` solves an instance with the native kernel (``DISCOPT_NATIVE_WARM_START``,
default 2) and pickles every node LP that was warm-started, with its exact start
basis (statuses and basic-variable order) and the in-tree pivot count.

``run`` re-solves each captured (LP, start) pair with ``solve_lp_warm_csc_py`` in
this process -- so any env knob read once per process (``DISCOPT_LP_*``) selects
the variant -- and reports total pivots, time, verdicts, and the ``DISCOPT_PROFILE``
counters. With no knob set it must reproduce the in-tree pivots exactly; the
script says how many it reproduced, so a harness that measures something else
cannot read as a baseline (CLAUDE.md §6).

    python -u .../issue1935_node_lp_pivot_harness.py capture --nl X.nl --out lps.pkl
    DISCOPT_PROFILE=1 python -u .../issue1935_node_lp_pivot_harness.py run lps.pkl
"""

from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def capture(nl: str, out: str, mode: str, tl: float, max_nodes: int) -> int:
    sys.path.insert(0, HERE)
    from issue1935_highs_node_lp_replay import capture as cap

    records, res = cap(nl, mode, tl, max_nodes)
    warm = [r for r in records if "start_col_status" in r]
    with open(out, "wb") as f:
        pickle.dump(warm, f)
    print(f"captured {len(warm)} warm-started node LPs of {len(records)} ({res.status})")
    return 0 if warm else 2


def run(path: str, flip: bool) -> int:
    import discopt._rust as _rust

    with open(path, "rb") as f:
        recs = pickle.load(f)
    if os.environ.get("DISCOPT_PROFILE"):
        _rust.profile_reset_py()
    tot_piv = 0
    tot_s = 0.0
    reproduced = 0
    verdict = collections.Counter()
    same_verdict = 0
    for r in recs:
        t0 = time.perf_counter()
        st, _x, obj, iters, _cs, _bv, _d, _ray = _rust.solve_lp_warm_csc_py(
            r["c"],
            int(r["m"]),
            int(r["n"]),
            r["col_ptr"].astype(np.int64),
            r["row_idx"].astype(np.int64),
            r["vals"],
            r["b"],
            r["l"],
            r["u"],
            start_col_status=r["start_col_status"].astype(np.int8),
            start_basic_vars=r["start_basic_vars"].astype(np.int64),
            warm_flip_boxed=flip,
        )
        tot_s += time.perf_counter() - t0
        tot_piv += iters
        reproduced += int(iters == r["n_pivots"])
        verdict[st] += 1
        same_verdict += int(st == r["status"])
    n = len(recs)
    print(
        f"lps={n} pivots={tot_piv} (in-tree {sum(r['n_pivots'] for r in recs)}) "
        f"time_s={tot_s:.3f} reproduced_in_tree_pivots={reproduced}/{n} "
        f"same_verdict={same_verdict}/{n} verdicts={dict(verdict)}",
        flush=True,
    )
    if os.environ.get("DISCOPT_PROFILE"):
        c = _rust.profile_counters_py()
        keys = (
            "DualWarmSolves",
            "DualDegeneratePivots",
            "DualBlandActivations",
            "DualDegenerateRunArms",
            "DualDegenerateStallBails",
            "DualCostPerturbAttempts",
            "DualCostPerturbAccepted",
            "DualCostPerturbRejected",
            "DualPrepAccept",
            "DualPrepRejectDualInf",
            "DualPrepRejectSingular",
            "EntryColsWarm",
            "Phase2Pivots",
            "DualRefactorizations",
        )
        print("  " + " ".join(f"{k}={c.get(k, 0)}" for k in keys), flush=True)
    return 0 if n else 2


def run_highs(path: str) -> int:
    """HiGHS (#1935 settings) on the same captured LPs, from the SAME start basis
    the kernel used -- isolating the engine from the quality of the start."""
    import highspy

    sys.path.insert(0, HERE)
    from issue1935_highs_node_lp_replay import _highs, _pass

    with open(path, "rb") as f:
        recs = pickle.load(f)
    st = highspy.HighsBasisStatus
    code = {0: st.kLower, 1: st.kBasic, 2: st.kUpper}
    h = _highs()
    tot_piv, tot_s, ok, rejected = 0, 0.0, 0, 0
    for r in recs:
        _pass(h, r)
        basis = highspy.HighsBasis()
        basis.col_status = [code[int(v)] for v in r["start_col_status"]]
        basis.row_status = [st.kLower] * int(r["m"])
        basis.valid = True
        t0 = time.perf_counter()
        if h.setBasis(basis) != highspy.HighsStatus.kOk:
            rejected += 1
        h.run()
        tot_s += time.perf_counter() - t0
        tot_piv += h.getInfo().simplex_iteration_count
        ok += int(h.getModelStatus() == highspy.HighsModelStatus.kOptimal)
    print(
        f"highs-from-kernel-start lps={len(recs)} pivots={tot_piv} time_s={tot_s:.3f} "
        f"optimal={ok} basis_rejected={rejected}",
        flush=True,
    )
    return 0 if recs else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--nl", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--mode", default="2", choices=["1", "2"])
    c.add_argument("--time-limit", type=float, default=30.0)
    c.add_argument("--max-nodes", type=int, default=1500)
    r = sub.add_parser("run")
    r.add_argument("path")
    r.add_argument("--no-flip", action="store_true", help="warm_flip_boxed off (mode 1)")
    hh = sub.add_parser("highs")
    hh.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "capture":
        return capture(a.nl, a.out, a.mode, a.time_limit, a.max_nodes)
    if a.cmd == "highs":
        return run_highs(a.path)
    return run(a.path, not a.no_flip)


if __name__ == "__main__":
    sys.exit(main())
