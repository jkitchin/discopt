"""Bisect which driver feature turns the #1141 node-hook run into a false optimal.

Injects driver option overrides through `_StdForm.lp_kwargs` (forwarded verbatim to
the binding) and reports the certified answer per arm. Reference optimum with node
cuts OFF: -0.10089619806602235.
"""
import os, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
import discopt.solvers.milp_simplex as ms
import discopt.solvers.oa as oa

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)
REF = -0.10089619806602235

OVERRIDES = {
    "none": {},
    "no_rcf": dict(reduced_cost_fixing=False),
    "no_heur": dict(heuristics=False),
    "no_sb": dict(strong_branch=False),
    "no_presolve": dict(presolve=False),
    "no_rootcuts": dict(root_cuts=0, cut_rounds=0, gmi_cuts=False),
    "all_off": dict(reduced_cost_fixing=False, heuristics=False, strong_branch=False,
                    presolve=False, root_cuts=0, cut_rounds=0, gmi_cuts=False),
}

_orig_marshal = ms._marshal_std_form
EXTRA = {}


def marshal(*a, **k):
    std = _orig_marshal(*a, **k)
    if EXTRA:
        std.lp_kwargs.update(EXTRA)
    return std


ms._marshal_std_form = marshal

for name in sys.argv[1:] or list(OVERRIDES):
    EXTRA.clear()
    EXTRA.update(OVERRIDES[name])
    for arm in ("off", "on"):
        os.environ["DISCOPT_OA_NODE_CUTS"] = "1" if arm == "on" else "0"
        m = portfolio2.build(**KW)
        r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                    time_limit=180, gap_tolerance=1e-4)
        ok = r.objective is not None and abs(r.objective - REF) <= 1e-9
        print(f"{name:12s} {arm:3s} status={str(r.status):10s} obj={r.objective!r} "
              f"bound={r.bound!r} {'OK' if ok else '<<< DIVERGED'}", flush=True)
