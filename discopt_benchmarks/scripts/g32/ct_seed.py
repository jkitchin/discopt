"""#1917 G3.2 step 1 fixture soundness: integer-feasible points of the ORIGINAL MTP model
(the optimum + random-objective MILP optima + continuous re-solves at fixed integers) must
validate as seeds with DISCOPT_RUST_MILP_COEF_TIGHTEN=1 exactly as with =0. usage: ct_seed.py <mps>"""
import os, sys
os.environ["DISCOPT_PROFILE"] = "1"
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R
assert b"MilpSeedRefused" in open(R.__file__, "rb").read(), "marker absent"
import numpy as np, scipy.sparse as sp, highspy
from discopt._rust import solve_milp_csc_py
from discopt.solvers.milp_simplex import _marshal_std_form
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.readModel(sys.argv[1]); lp = h.getLp()
A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()
rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_); INF = 1e30
eq = (rl == ru); up = (~eq) & (ru < INF); lo = (~eq) & (rl > -INF)
A_ub = sp.vstack([A[up], -A[lo]]).tocsr(); b_ub = np.r_[ru[up], -rl[lo]]
cl, cu = np.array(lp.col_lower_), np.array(lp.col_upper_)
bounds = [(None if a <= -INF else a, None if b >= INF else b) for a, b in zip(cl, cu)]
integ = np.array([1 if t == highspy.HighsVarType.kInteger else 0 for t in lp.integrality_])
c0 = np.array(lp.col_cost_); n = len(c0); I = integ > 0
std = _marshal_std_form(c0, A_ub, b_ub, A[eq].tocsr(), ru[eq], bounds, integ)
def viol(x):
    act = A @ x
    v = max(np.max(np.maximum(rl - act, 0)), np.max(np.maximum(act - ru, 0)), np.max(np.maximum(cl - x, 0)), np.max(np.maximum(x - cu, 0)))
    return max(v, np.max(np.abs(x[I] - np.round(x[I]))))
def solve(cost, fix=None):
    g = highspy.Highs(); g.setOptionValue("output_flag", False); g.setOptionValue("threads", 1)
    g.setOptionValue("time_limit", 5.0); g.passModel(lp); g.changeColsCost(n, np.arange(n), cost)
    if fix is not None:
        idx = np.where(I)[0]; g.changeColsBounds(len(idx), idx, fix[idx], fix[idx])
        g.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kContinuous] * len(idx)))
    g.run(); s = g.getSolution()
    if not s.value_valid: return None
    x = np.array(s.col_value);
    if fix is not None: x[I] = fix[I]
    return x
rng = np.random.default_rng(19176332)
pts = []
xo = solve(c0); assert xo is not None and viol(xo) < 1e-6; pts.append(("opt", xo))
print("optimum obj", c0 @ xo + lp.offset_, flush=True)
for k in range(30):
    x = solve(rng.normal(size=n))
    if x is None: continue
    x[I] = np.round(x[I]); pts.append((f"rand{k}", x))
    for r in range(3):
        xf = solve(rng.normal(size=n), fix=x)
        if xf is not None: pts.append((f"rand{k}.c{r}", xf))
checked = 0; ok_both = 0
for name, x in pts:
    v = viol(x)
    if v > 1e-7:  # only points feasible in the ORIGINAL model are evidence
        print(f"skip {name} viol {v:.2e}"); continue
    res = {}
    for arm in ("0", "1"):
        os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = arm; R.profile_reset_py()
        solve_milp_csc_py(std.c, std.m, n + std.m, std.col_ptr, std.row_idx, std.vals, std.b, std.lb, std.ub,
            std.int_cols, n, 0.0, 2, 1e-4, time_limit_s=10.0, initial_incumbent=np.ascontiguousarray(x, dtype=float))
        pc = R.profile_counters_py()
        res[arm] = (pc["MilpSeedValidated"], pc["MilpSeedRefused"], pc["MilpCoefTightenCoefs"])
        assert res[arm][0] + res[arm][1] == 1, res
    assert res["1"][2] > 0 and res["0"][2] == 0, res
    assert res["0"][0] == 1, f"OFF refused a feasible point {name}: {res}"
    assert res["1"][0] == 1, f"ON refused a point feasible in the original {name} (viol {v:.1e}): {res}"
    checked += 1; ok_both += 1
print(f"PROBE_DONE seeds_checked={checked} validated_both={ok_both} of {len(pts)} points", flush=True)
sys.exit(0 if checked >= 20 else 1)
