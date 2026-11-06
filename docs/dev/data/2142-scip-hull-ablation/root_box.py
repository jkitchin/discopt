"""#2142: discopt's root FBBT box on the homog spec vs SCIP's presolved bounds.
Reads presolved_homog.cip (SCIP presolve, from presolve_dump.py) for the surviving
variables' global bounds and the fixed/aggregated ones, runs the kernel's propagator
(propagate_spec_py, the per-node FBBT) on the spec's global box with the round cap the
tree uses (15) and to convergence (100000), and counts: columns SCIP fixed that discopt
also fixes, and the sum of box widths. Executed-assertion count at the end."""
import re, sys
import numpy as np
import discopt._rust as R
assert R.__file__.startswith("/home/user/discopt/python"), R.__file__
n_assert = 1
SPEC = "/home/user/discopt/docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz"
d = dict(np.load(SPEC))
n = int(d["n_cols"]); n_orig = int(d["n_orig"])
lo0, hi0 = d["global_lo"].astype(float), d["global_hi"].astype(float)
txt = open("presolved_homog.cip").read()
scip_lo, scip_hi = lo0.copy(), hi0.copy()
fixed_scip = {}
for mm in re.finditer(r"<t_x(\d+)>: obj=\S+, global bounds=\[([^,]+),([^\]]+)\]", txt):
    j = int(mm.group(1)); a, b = float(mm.group(2)), float(mm.group(3))
    scip_lo[j], scip_hi[j] = a, b
for mm in re.finditer(r"\] <t_x(\d+)>:.*fixed:(\S+)", txt):
    j = int(mm.group(1)); v = float(mm.group(2)); fixed_scip[j] = v; scip_lo[j] = scip_hi[j] = v
aggr = {}
for mm in re.finditer(r"\] <t_x(\d+)>:.*(?:aggregated|negated): (.*)", txt):
    aggr[int(mm.group(1))] = mm.group(2)
print(f"SCIP presolve: fixed {len(fixed_scip)} cols, aggregated/negated {len(aggr)}")
kw = {k: d[k] for k in ("n_cols","n_orig","c","integrality","global_lo","global_hi","fixed_row_ptr","fixed_cols","fixed_coeffs","fixed_rhs","term_kind","term_i","term_j","term_out","term_p","term_coeff","term_cst","blf_w","blf_a_ptr","blf_a_cols","blf_a_coeffs","blf_a_const","blf_b_ptr","blf_b_cols","blf_b_coeffs","blf_b_const","obbt_candidates")}
for k in ("n_cols","n_orig"): kw[k] = int(kw[k])
for k in list(kw):
    if isinstance(kw[k], np.ndarray):
        kw[k] = kw[k].astype(np.int64) if kw[k].dtype.kind in "iub" else kw[k].astype(float)
for rounds in (15, 100000):
    live, lo, hi = R.propagate_spec_py(**kw, lo=lo0.copy(), hi=hi0.copy(), cutoff=None, rounds=rounds)
    lo, hi = np.asarray(lo), np.asarray(hi)
    assert live, "root box empty?!"; n_assert += 1
    w = np.clip(hi - lo, 0, None)
    fixed_us = [j for j in range(n) if w[j] <= 1e-7]
    both = [j for j in fixed_scip if w[j] <= 1e-7]
    print(f"discopt FBBT rounds={rounds}: fixed {len(fixed_us)} cols; of SCIP's {len(fixed_scip)} fixed, discopt fixes {len(both)}; "
          f"sum width orig {w[:n_orig].sum():.3f} (global {np.clip(hi0-lo0,0,None)[:n_orig].sum():.3f}); "
          f"cols tightened vs global {int(((lo>lo0+1e-9)|(hi<hi0-1e-9)).sum())}")
    miss = [j for j in fixed_scip if w[j] > 1e-7]
    print("  SCIP-fixed, discopt not:", [(j, round(lo[j],4), round(hi[j],4), fixed_scip[j]) for j in miss[:60]])
    # SCIP bounds tighter than ours by > 1e-6 among surviving (non-aggregated) cols
    tighter = [(j, round(lo[j],4), round(hi[j],4), scip_lo[j], scip_hi[j]) for j in range(n) if j not in aggr and j not in fixed_scip and (scip_lo[j] > lo[j] + 1e-6 or scip_hi[j] < hi[j] - 1e-6)]
    print(f"  SCIP tighter on {len(tighter)} surviving cols:", tighter[:40])
print(f"executed assertions: {n_assert}")
