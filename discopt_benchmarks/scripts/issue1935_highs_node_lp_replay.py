"""#1935: the in-house simplex vs HiGHS on the SAME node LPs, same machine.

#1935 measured HiGHS warm-started from the parent against this kernel's cold
in-house simplex, on another machine. This replays one tree's node-LP sequence
through both engines:

1. Solve an instance normally with the native kernel at ``DISCOPT_NATIVE_WARM_START``
   (default 2), attaching the kernel's ``node_observer``. Every solved node hands
   over its assembled LP (CSC ``[A | I] z = b``, bounds, cost), the basis it was
   started from, and the in-house result: status, pivots, objective, and
   ``lp_s`` (the simplex call alone, timed in Rust).
2. Replay those LPs in tree order through highspy, configured as #1935 did:
   silent, one thread, presolve off, serial dual simplex, one ``Highs`` per tree.
   HiGHS warm-starts each child from ITS OWN optimal basis for the parent (or,
   when the parent did not solve, from what the parent was started from -- the
   kernel's rule). Each node is also solved cold in HiGHS, as a reference.
   HiGHS time is ``setBasis`` + ``run`` (``setBasis`` factorizes), timed from
   Python, so it carries a few microseconds of binding overhead the Rust timing
   does not.

Reported per instance and pooled: median and total LP time, median pivots, and
status / objective agreement between the engines on every node. Exits non-zero
when it compared nothing (CLAUDE.md §6).

    python -u discopt_benchmarks/scripts/issue1935_highs_node_lp_replay.py \\
        --minlplib ~/.cache/discopt/minlplib/current --instances nvs20,nous2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

INF = 1e20  # the Rust LP layer's infinity sentinel


def capture(path: str, mode: str, tl: float, max_nodes: int) -> tuple[list[dict], object]:
    os.environ["DISCOPT_NATIVE_WARM_START"] = mode
    import discopt._rust as _rust
    from discopt.modeling.core import from_nl

    real = _rust.solve_spatial_tree_py
    assert "node_observer" in (real.__text_signature__ or ""), "stale build"
    records: list[dict] = []

    def observer(rec: dict) -> None:
        if len(records) < max_nodes:
            records.append(
                {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
            )

    def wrapped(*a, **kw):
        kw["node_observer"] = observer
        return real(*a, **kw)

    _rust.solve_spatial_tree_py = wrapped
    try:
        res = from_nl(path).solve(time_limit=tl)
    finally:
        _rust.solve_spatial_tree_py = real
    return records, res


def _highs(threads_one: bool = True):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("simplex_strategy", 1)  # serial dual
    if threads_one:
        h.setOptionValue("threads", 1)
    return h


def _pass(h, r: dict) -> bool:
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = int(r["n"]), int(r["m"])
    lp.col_cost_ = r["c"]
    lp.col_lower_ = np.where(r["l"] <= -INF, -np.inf, r["l"])
    lp.col_upper_ = np.where(r["u"] >= INF, np.inf, r["u"])
    lp.row_lower_ = r["b"]
    lp.row_upper_ = r["b"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = r["col_ptr"].astype(np.int32)
    lp.a_matrix_.index_ = r["row_idx"].astype(np.int32)
    lp.a_matrix_.value_ = r["vals"]
    st = h.passModel(lp)
    # kWarning is HiGHS dropping |a| <= small_matrix_value (1e-9), #1935 caveat 1:
    # its point can then violate the assembled LP. Counted, not refused.
    assert st != highspy.HighsStatus.kError, st
    return st == highspy.HighsStatus.kWarning


def replay(records: list[dict]) -> list[dict]:
    import highspy

    warm, cold = _highs(), _highs()
    end_basis: dict[int, object] = {}  # HiGHS's optimal basis per node id
    start_basis: dict[int, object] = {}  # what HiGHS started each node from
    out = []
    for r in sorted(records, key=lambda r: r["id"]):
        start = None
        p = r["parent"]
        if p is not None:
            start = end_basis.get(p, start_basis.get(p))
        dropped = _pass(warm, r)
        t0 = time.perf_counter()
        if start is not None:
            warm.setBasis(start)
        warm.run()
        dt = time.perf_counter() - t0
        st = warm.getModelStatus()
        info = warm.getInfo()
        rec = {
            "id": r["id"],
            "d_status": r["status"],
            "d_s": r["lp_s"],
            "d_piv": r["n_pivots"],
            "d_obj": r["obj"],
            "d_warm": r["warm_started"],
            "h_status": warm.modelStatusToString(st),
            "h_s": dt,
            "h_piv": info.simplex_iteration_count,
            "h_obj": info.objective_function_value,
            "h_warm": start is not None,
            "h_dropped_entries": dropped,
        }
        start_basis[r["id"]] = start
        if st == highspy.HighsModelStatus.kOptimal:
            end_basis[r["id"]] = warm.getBasis()
        _pass(cold, r)
        cold.clearSolver()
        t0 = time.perf_counter()
        cold.run()
        rec["hc_s"] = time.perf_counter() - t0
        rec["hc_piv"] = cold.getInfo().simplex_iteration_count
        out.append(rec)
    return out


def _agree(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * (1.0 + max(abs(a), abs(b)))


def report(name: str, rows: list[dict]) -> dict:
    both = [r for r in rows if r["d_status"] == "optimal" and r["h_status"] == "Optimal"]
    s = {
        "name": name,
        "nodes": len(rows),
        "both_optimal": len(both),
        "obj_agree": sum(_agree(r["d_obj"], r["h_obj"]) for r in both),
        "h_dropped": sum(r["h_dropped_entries"] for r in rows),
        "d_only_fail": sum(r["d_status"] != "optimal" and r["h_status"] == "Optimal" for r in rows),
        "h_only_fail": sum(r["d_status"] == "optimal" and r["h_status"] != "Optimal" for r in rows),
        "d_med_ms": 1e3 * statistics.median(r["d_s"] for r in rows),
        "h_med_ms": 1e3 * statistics.median(r["h_s"] for r in rows),
        "hc_med_ms": 1e3 * statistics.median(r["hc_s"] for r in rows),
        "d_tot_s": sum(r["d_s"] for r in rows),
        "h_tot_s": sum(r["h_s"] for r in rows),
        "hc_tot_s": sum(r["hc_s"] for r in rows),
        "d_med_piv": statistics.median(r["d_piv"] for r in rows),
        "h_med_piv": statistics.median(r["h_piv"] for r in rows),
        "hc_med_piv": statistics.median(r["hc_piv"] for r in rows),
    }
    print(
        f"{name:22s} nodes={s['nodes']:5d} both-opt={s['both_optimal']:5d} "
        f"obj-agree={s['obj_agree']:5d} discopt-only-fail={s['d_only_fail']:4d} "
        f"highs-only-fail={s['h_only_fail']:4d} highs-dropped-a={s['h_dropped']:4d} | "
        f"median ms discopt {s['d_med_ms']:.3f} "
        f"highs {s['h_med_ms']:.3f} (cold {s['hc_med_ms']:.3f}) | total s "
        f"{s['d_tot_s']:.2f} / {s['h_tot_s']:.2f} (cold {s['hc_tot_s']:.2f}) | median piv "
        f"{s['d_med_piv']:.0f} / {s['h_med_piv']:.0f} (cold {s['hc_med_piv']:.0f})",
        flush=True,
    )
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minlplib", required=True, help="MINLPLib download with nl/")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--mode", default="2", choices=["0", "1", "2"])
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--max-nodes", type=int, default=1500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print(f"load {os.getloadavg()}", flush=True)
    summaries = []
    for name in a.instances.split(","):
        path = os.path.join(os.path.expanduser(a.minlplib), "nl", f"{name}.nl")
        records, res = capture(path, a.mode, a.time_limit, a.max_nodes)
        if not records:
            print(f"{name:22s} no node LPs captured (route {res.algorithm_route})", flush=True)
            continue
        rows = replay(records)
        summaries.append(report(name, rows))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps({"name": name, "rows": rows}) + "\n")
    if not summaries:
        print("NOTHING COMPARED", flush=True)
        return 2
    tot = {k: sum(s[k] for s in summaries) for k in ("nodes", "both_optimal", "obj_agree")}
    print(f"\npooled: {tot}")
    for k in ("d_tot_s", "h_tot_s", "hc_tot_s"):
        print(f"  {k}: {sum(s[k] for s in summaries):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
