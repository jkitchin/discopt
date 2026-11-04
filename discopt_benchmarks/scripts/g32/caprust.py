"""Capture the first solve_milp_csc_py call of a GDPlib solve (exact Rust-MILP input).
usage: caprust.py <instance> <gdp_method> <out.pkl>"""
import os, sys, pickle, inspect
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
sys.path.insert(0, W + "/discopt_benchmarks"); sys.path.insert(0, W + "/discopt_benchmarks/scripts")
import numpy as np
import discopt._rust as R
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
real = R.solve_milp_csc_py
NAMES = ["c", "m", "n", "col_ptr", "row_idx", "vals", "b", "lb", "ub", "integer_cols", "n_struct", "obj_const",
         "max_nodes", "gap_tol", "tol", "root_cuts", "cut_rounds", "gmi_cuts", "cut_select", "node_cuts", "max_pool_cuts",
         "heuristics", "presolve", "strong_branch", "node_propagation", "reduced_cost_fixing", "sb_max_cands",
         "sb_node_budget", "initial_incumbent", "time_limit_s", "root_cut_time_s", "root_cut_prune", "debug_hook",
         "abs_gap_tol", "max_open_nodes"]
def wrapped(*a, **k):
    d = dict(zip(NAMES, a)); d.update(k)
    d["had_debug_hook"] = d.pop("debug_hook", None) is not None
    d = {kk: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for kk, v in d.items()}
    pickle.dump(d, open(sys.argv[3], "wb"))
    print("captured", sys.argv[1], sys.argv[2], "m", d["m"], "n", d["n"], "ns", d["n_struct"], "ints", len(d["integer_cols"]), "hook", d["had_debug_hook"], flush=True)
    os._exit(0)
R.solve_milp_csc_py = wrapped
conv = convert_pyomo_gdp(_spec(sys.argv[1]).builder())
conv.model.solve(gdp_method=sys.argv[2], time_limit=60)
print("no Rust MILP call", sys.argv[1], sys.argv[2], flush=True)
