"""Budget sweep on hartman_3 (issue #1036), plus the design-size each budget picks.

Prints one row per budget and exits non-zero if it produced no rows (CLAUDE.md §6).
"""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "tests"))
import numpy as np
import discopt.solvers.surrogate as S
print("loaded:", S.__file__, flush=True)
from support import direct_testfuncs as tfs

def design_size(n_vars, max_evals, n_initial=None):
    if n_initial is None:
        d = max(n_vars + 2, min(10 * n_vars, max(1, max_evals // 2)))
    else:
        d = int(n_initial)
    return max(1, min(d, max_evals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="hartman_3")
    ap.add_argument("--budgets", default="46,60,80,100,120,150,200")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-initial", default="", help="fixed n_initial, or blank for the default rule")
    args = ap.parse_args()
    tf = tfs.get(args.func)
    n_init = int(args.n_initial) if args.n_initial else None
    rows = 0
    print(f"{'budget':>7} {'design':>7} {'seed':>5} {'objective':>12} {'rel_err':>10}  first@1e-2")
    for b in [int(v) for v in args.budgets.split(",")]:
        for s in [int(v) for v in args.seeds.split(",")]:
            trace = []
            model, _ = tfs.build_model(tf)
            kw = dict(max_evals=b, time_limit=600.0, seed=s,
                      acquisition_optimizer="multistart",
                      on_evaluation=lambda n, v: trace.append((n, v)))
            if n_init is not None:
                kw["n_initial"] = n_init
            r = S.solve_surrogate(model, **kw)
            assert trace, "on_evaluation never fired - this probe measured nothing"
            first = next((n for n, v in trace if v is not None and tf.relative_error(v) <= 1e-2), None)
            print(f"{b:7d} {design_size(tf.n, b, n_init):7d} {s:5d} {r.objective:12.6f} "
                  f"{tf.relative_error(r.objective):10.4e}  {first}", flush=True)
            rows += 1
    print(f"executed rows: {rows}")
    if rows == 0:
        sys.exit(1)

main()
