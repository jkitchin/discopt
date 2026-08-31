"""Why does `clay0303hfsg`'s dual bound collapse under fractional separation? (#1141)

Instruments the master: node count, LP row growth, separator call/row counts, and
time spent inside the node separator. Prints an executed-run count (§6).
"""
import os, sys, time
import numpy as np
from discopt.modeling.core import from_nl
import discopt.solvers.milp_simplex as ms

NAME = sys.argv[1] if len(sys.argv) > 1 else "clay0303hfsg"
DIR = sys.argv[2] if len(sys.argv) > 2 else "python/tests/data/minlplib_nl"

_orig = ms.solve_milp_with_lazy_cuts
info = {}


def wrapped(**kw):
    A = kw.get("A_ub")
    info["rows_in"] = 0 if A is None else A.shape[0]
    info["cols"] = np.asarray(kw["c"]).shape[0]
    info["sep_time"] = 0.0
    nc = kw.get("node_callback")
    if nc is not None:
        def timed(x):
            t = time.perf_counter()
            try:
                return nc(x)
            finally:
                info["sep_time"] += time.perf_counter() - t
        kw["node_callback"] = timed
    r = _orig(**kw)
    info["milp_status"] = str(r.status)
    info["milp_nodes"] = r.node_count
    info["milp_bound"] = r.bound
    info["milp_obj"] = r.objective
    info["milp_wall"] = r.wall_time
    info["stats"] = dict(r.callback_stats or {})
    return r


ms.solve_milp_with_lazy_cuts = wrapped
import discopt.solvers.oa as oa
oa.solve_milp_with_lazy_cuts = wrapped

runs = 0
for arm, cuts, rounds, cap in [("off", "0", "2", "500"), ("on", "1", "2", "500")]:
    os.environ["DISCOPT_OA_NODE_CUTS"] = cuts
    os.environ["DISCOPT_OA_NODE_CUT_ROUNDS"] = rounds
    os.environ["DISCOPT_OA_NODE_CUT_CAP"] = cap
    info.clear()
    m = from_nl(f"{DIR}/{NAME}.nl")
    t = time.perf_counter()
    r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                time_limit=30, gap_tolerance=1e-4)
    runs += 1
    st = info.get("stats", {})
    print(f"{NAME} {arm}: outer status={r.status} obj={r.objective!r} bound={r.bound!r} "
          f"wall={time.perf_counter()-t:.2f}")
    print(f"    master: status={info.get('milp_status')} nodes={info.get('milp_nodes')} "
          f"bound={info.get('milp_bound')!r} wall={info.get('milp_wall'):.2f}")
    print(f"    LP: {info.get('rows_in')} rows x {info.get('cols')} cols; "
          f"separator time {info.get('sep_time', 0.0):.2f}s")
    print(f"    stats: mipsol={st.get('mipsol_calls')} lazy_cuts={st.get('lazy_cuts')} "
          f"requeues={st.get('lazy_requeues')} mipnode={st.get('mipnode_calls')} "
          f"node_cuts={st.get('driver_node_cuts')}", flush=True)
print(f"\nEXECUTED RUNS: {runs}")
sys.exit(1 if runs == 0 else 0)
