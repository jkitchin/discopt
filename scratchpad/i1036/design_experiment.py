"""Entry experiment for issue #1036, H2: is the initial design oversized?

Holds ``max_evals`` fixed and varies ONLY ``n_initial``, so the design-size rule
is the single independent variable. Reports, per (function, arm): how many seeds
reached 1e-2 relative error inside the budget, the median evaluation at which
they first did, and the median final relative error.

Prints an executed-run count and exits non-zero if it is zero (CLAUDE.md §6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python", "tests"))

import numpy as np  # noqa: E402

#: n_initial as a function of the dimension. ``None`` = whatever the shipped rule
#: does -- so the "shipped" arm tracks the LIVE code. The committed results were
#: taken before #1036, when that was ``max(n+2, min(10n, max_evals // 2))``; on
#: this tree it now resolves to ``2(n+1)`` and duplicates that arm. To reproduce
#: the "shipped" column, restore the old expression in ``solve_surrogate`` first.
ARMS: dict[str, object] = {
    "shipped": None,
    "n+2": lambda n: n + 2,
    "2(n+1)": lambda n: 2 * (n + 1),
    "(n+1)(n+2)/2": lambda n: (n + 1) * (n + 2) // 2,
    "5n": lambda n: 5 * n,
    "10n": lambda n: 10 * n,
}


def _dump_rows(rows: list[dict], path: str) -> None:
    """One row per line: the same data as ``indent=1``, ~10x fewer diff lines."""
    body = ",\n ".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with open(path, "w") as fh:
        fh.write("[" + body + "\n]\n")


def _one(job):
    import discopt.solvers.surrogate as S
    from support import direct_testfuncs as tfs

    func, arm, seed, budget, tol, family = job
    tf = tfs.get(func)
    rule = ARMS[arm]
    trace: list[tuple[int, float | None]] = []
    model, _ = tfs.build_model(tf)
    kw = dict(
        max_evals=budget,
        time_limit=3600.0,
        seed=seed,
        acquisition_optimizer="multistart",
        surrogate=family,
        on_evaluation=lambda k, v: trace.append((k, v)),
    )
    if rule is not None:
        kw["n_initial"] = int(rule(tf.n))
    r = S.solve_surrogate(model, **kw)
    if not trace:
        raise AssertionError(f"on_evaluation never fired for {job} - measured nothing")
    first = next((k for k, v in trace if v is not None and tf.relative_error(v) <= tol), None)
    return {
        "func": func,
        "family": family,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "n_initial": None if rule is None else int(rule(tf.n)),
        "first": first,
        "rel_err": tf.relative_error(r.objective) if r.objective is not None else None,
        "module": S.__file__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--funcs",
        default="branin,six_hump_camel,ackley_2,hartman_3,goldstein_price,hartman_6,rastrigin_2,shubert",
    )
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--family", default="rbf", choices=("rbf", "kriging"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "design_experiment.json"))
    args = ap.parse_args()

    funcs = args.funcs.split(",")
    arms = args.arms.split(",")
    jobs = [
        (f, a, s, args.budget, args.tol, args.family)
        for f in funcs
        for a in arms
        for s in range(args.seeds)
    ]
    print(f"{len(jobs)} runs: {len(funcs)} functions x {len(arms)} arms x {args.seeds} seeds "
          f"at max_evals={args.budget}, surrogate={args.family}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(_one, jobs), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} done", flush=True)

    _dump_rows(rows, args.out)

    # -- report ---------------------------------------------------------------
    print()
    hdr = f"{'function':<17}{'arm':<15}{'n_init':>7}{'reached':>9}{'med first':>11}{'med relerr':>12}"
    print(hdr)
    print("-" * len(hdr))
    for f in funcs:
        for a in arms:
            sub = [r for r in rows if r["func"] == f and r["arm"] == a]
            if not sub:
                continue
            hits = [r["first"] for r in sub if r["first"] is not None]
            med = f"{np.median(hits):.0f}" if hits else "-"
            errs = [r["rel_err"] for r in sub if r["rel_err"] is not None]
            n_init = sub[0]["n_initial"]
            print(
                f"{f:<17}{a:<15}{'auto' if n_init is None else n_init:>7}"
                f"{f'{len(hits)}/{len(sub)}':>9}{med:>11}"
                f"{(np.median(errs) if errs else float('nan')):>12.3e}"
            )
        print()

    print(f"executed runs: {len(rows)}")
    print("module under test:", rows[0]["module"] if rows else "NONE")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
