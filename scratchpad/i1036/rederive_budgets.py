"""Re-derive the convergence panel's budgets (issue #1036, item 1).

With the initial design no longer a function of ``max_evals`` a run at budget B is
the run at any smaller budget, continued -- so a single trace per (function, seed)
at a large budget yields the evaluation at which that seed first reaches the
tolerance for EVERY budget at once. That is the property the old docstring's
"first reached at k, so B has headroom" argument assumed and did not have.

Reports, per function, the evaluation at which the LAST of the k seeds first
reaches the tolerance -- i.e. the smallest budget at which k of k seeds pass.
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python", "tests"))
import numpy as np  # noqa: E402


def _one(job):
    import discopt.solvers.surrogate as S
    from support import direct_testfuncs as tfs
    func, seed, budget, tol = job
    tf = tfs.get(func)
    trace: list[tuple[int, float | None]] = []
    model, _ = tfs.build_model(tf)
    S.solve_surrogate(model, max_evals=budget, time_limit=3600.0, seed=seed,
                      acquisition_optimizer="multistart",
                      on_evaluation=lambda k, v: trace.append((k, v)))
    if not trace:
        raise AssertionError(f"on_evaluation never fired for {job}")
    first = next((k for k, v in trace if v is not None and tf.relative_error(v) <= tol), None)
    return {"func": func, "seed": seed, "first": first, "budget": budget}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", default="branin,six_hump_camel,ackley_2,hartman_3,goldstein_price")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "rederive_budgets.json"))
    args = ap.parse_args()
    funcs = args.funcs.split(",")
    jobs = [(f, s, args.budget, args.tol) for f in funcs for s in range(args.seeds)]
    print(f"{len(jobs)} traces at max_evals={args.budget}, tol={args.tol}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    json.dump(rows, open(args.out, "w"), indent=1)

    print(f"\n{'function':<18}{'per-seed first-reach':<46}{'k/k budget':>11}")
    print("-" * 75)
    for f in funcs:
        sub = sorted((r for r in rows if r["func"] == f), key=lambda r: r["seed"])
        hits = [r["first"] for r in sub]
        shown = ", ".join("-" if h is None else str(h) for h in hits)
        worst = "never" if any(h is None for h in hits) else str(max(hits))
        print(f"{f:<18}{shown:<46}{worst:>11}")
    print(f"\nexecuted traces: {len(rows)}")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
