"""#2142: apply arm_aggr.rewrite (the 106 expressible aggregations) to the captured homog
spec and save it, so SCIP can be run on the already-aggregated model with presolve off.
Prints an executed-assertion count (CLAUDE.md §6)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "arm_aggr.py")).read()
src = src[: src.index("def check_original")]
sys.argv = ["arm_aggr.py", "aggr", "0"]
g = {"__file__": os.path.join(HERE, "..", "arm_aggr.py"), "__name__": "aggr_lib"}
exec(compile(src, "arm_aggr.py", "exec"), g)
d = dict(np.load(sys.argv_out if False else os.path.join(HERE, "..", "..", "2142-exact-hull", "scip", "cstr_homog_spec.npz")))
n0 = int(d["n_cols"])
k, expand = g["rewrite"](dict(d))
fixed = int(np.sum((np.asarray(k["global_lo"]) == 0) & (np.asarray(k["global_hi"]) == 0)))
fixed0 = int(np.sum((d["global_lo"] == 0) & (d["global_hi"] == 0)))
n_assert = 0
assert int(k["n_cols"]) == n0; n_assert += 1
assert fixed > fixed0, (fixed, fixed0); n_assert += 1
print("stats", dict(g["STATS"]), "fixed-at-0 cols", fixed0, "->", fixed, "rows", len(d["fixed_rhs"]), "->", len(k["fixed_rhs"]))
np.savez(os.path.join(HERE, "cstr_homog_aggr_spec.npz"), **{a: np.asarray(v) for a, v in k.items()})
print("executed assertions:", n_assert)
