"""Capture the Rust-MILP input milp_simplex.solve_milp builds for each in-repo MILP fixture."""
import os, sys, gzip, pickle, tempfile
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import numpy as np, scipy.sparse as sp, highspy
import discopt._rust as R
from discopt.solvers import milp_simplex
D = W + "/python/tests/data/"
NAMES = ["c", "m", "n", "col_ptr", "row_idx", "vals", "b", "lb", "ub", "integer_cols", "n_struct", "obj_const",
         "max_nodes", "gap_tol", "tol", "root_cuts", "cut_rounds", "gmi_cuts", "cut_select", "node_cuts", "max_pool_cuts",
         "heuristics", "presolve", "strong_branch", "node_propagation", "reduced_cost_fixing", "sb_max_cands",
         "sb_node_budget", "initial_incumbent", "time_limit_s", "root_cut_time_s", "root_cut_prune", "debug_hook",
         "abs_gap_tol", "max_open_nodes"]

class Captured(BaseException): pass
cap = {}
def wrapped(*a, **k):
    d = dict(zip(NAMES, a)); d.update(k); d["had_debug_hook"] = d.pop("debug_hook", None) is not None
    cap["d"] = {kk: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for kk, v in d.items()}
    raise Captured()
R.solve_milp_csc_py = wrapped
def from_highs(path):
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    if path.endswith(".gz"):
        tf = tempfile.NamedTemporaryFile(suffix=".mps", delete=False); tf.write(gzip.open(path).read()); tf.close(); path = tf.name
    h.readModel(path); lp = h.getLp(); INF = 1e30
    A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()
    rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_)
    eq = (rl == ru); up = (~eq) & (ru < INF); lo = (~eq) & (rl > -INF)
    cl, cu = np.array(lp.col_lower_), np.array(lp.col_upper_)
    return dict(c=np.array(lp.col_cost_), A_ub=sp.vstack([A[up], -A[lo]]).tocsr(), b_ub=np.r_[ru[up], -rl[lo]],
                A_eq=A[eq].tocsr(), b_eq=ru[eq], bounds=[(None if a <= -INF else a, None if b >= INF else b) for a, b in zip(cl, cu)],
                integrality=np.array([1 if t == highspy.HighsVarType.kInteger else 0 for t in lp.integrality_]))
def from_master(path):
    z = np.load(path)
    mk = lambda p: sp.csr_matrix((z[p + "_data"], z[p + "_indices"], z[p + "_indptr"]), shape=tuple(z[p + "_shape"]))
    return dict(c=z["c"], A_ub=mk("A_ub"), b_ub=z["b_ub"], A_eq=mk("A_eq"), b_eq=z["b_eq"],
                bounds=[(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(z["lb"], z["ub"])],
                integrality=z["integrality"])
def from_rowform(path):
    z = np.load(path); A = sp.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]), shape=tuple(z["shape"]))
    rl, ru = z["row_lo"], z["row_hi"]; eq = rl == ru; up = (~eq) & np.isfinite(ru); lo = (~eq) & np.isfinite(rl)
    return dict(c=z["c"], A_ub=sp.vstack([A[up], -A[lo]]).tocsr(), b_ub=np.r_[ru[up], -rl[lo]], A_eq=A[eq].tocsr(), b_eq=ru[eq],
                bounds=[(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(z["col_lo"], z["col_hi"])],
                integrality=z["is_int"].astype(int))
items = [("inrepo__i1183", from_highs, W + "/discopt_benchmarks/data/lp_milp/i1183.mps"),
         ("inrepo__clay0203_master", from_highs, D + "issue2069/clay0203_master.mps.gz"),
         ("inrepo__highs_presolve_cycle_1671", from_highs, D + "highs_presolve_cycle_1671.mps.gz"),
         ("inrepo__issue1229_milp", from_rowform, D + "issue1229_issue2388_milp.npz")]
for f in ("fac2", "rsyn0830m", "tls2"):
    items.append((f"inrepo__{f}_master0", from_master, D + f"oa_masters/{f}_master0.npz"))
for name, fn, path in items:
    kw = fn(path); cap.clear()
    try:
        milp_simplex.solve_milp(time_limit=60.0, **kw)
    except Captured:
        pass
    assert "d" in cap, f"no Rust call for {name}"
    d = cap["d"]
    if name == "inrepo__issue1229_milp":
        d["obj_const"] = d.get("obj_const", 0.0)
    pickle.dump(d, open(f"{name}.pkl", "wb"))
    print("captured", name, "m", d["m"], "ns", d["n_struct"], "ints", len(d["integer_cols"]), flush=True)
print("CAP_INREPO_DONE")
