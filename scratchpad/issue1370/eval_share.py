"""#1370 Part B, second entry measurement: what share of a SOLVE is evaluation?

The first experiment says one vectorised pass over K identical blocks beats the
tape by 5-18x on the derivatives. That is only worth building if the
derivatives are a large share of the solve. The issue measures 48% on its `.nl`
path; discopt's path is POUNCE's Rust tape driven by POUNCE's own IPM, and the
share there has to be measured, not carried over — a 17x on 8% of the wall is
an 8% solve.

Two independent instruments, because one of them measuring nothing is the
failure mode this is written against:

  * `discopt._timing` buckets (`rust` is the tape evaluator's own bucket,
    `pounce` the enclosing IPM's), and
  * POUNCE's own `n_*_evals` counters multiplied by this harness's directly
    measured per-call costs.

Run:  python -u scratchpad/issue1370/eval_share.py --ks 8,32
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
from discopt import _timing
from discopt.solvers.nlp_pounce import solve_nlp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from block_eval_entry import make_block_model  # noqa: E402


def measure(K: int, steps: int, dim: int, reps: int) -> dict:
    from discopt._tape_nlp_evaluator import make_evaluator

    model = make_block_model(K, steps, dim)
    ev = make_evaluator(model)
    n, m = ev.n_variables, ev.n_constraints
    print(f"\n=== K={K} steps={steps} dim={dim}: n={n} m={m} ===", flush=True)

    rng = np.random.default_rng(0)
    x0 = rng.normal(size=n) * 0.1

    # Per-call costs, measured here rather than assumed.
    lam = rng.normal(size=m)
    for fn, label in (
        (lambda: ev.evaluate_constraints(x0), "g"),
        (lambda: ev.evaluate_jacobian_values(x0), "J"),
        (lambda: ev.evaluate_hessian_values(x0, 1.0, lam), "H"),
    ):
        fn()
    per_call = {}
    for label, fn in (
        ("g", lambda: ev.evaluate_constraints(x0)),
        ("grad", lambda: ev.evaluate_gradient(x0)),
        ("J", lambda: ev.evaluate_jacobian_values(x0)),
        ("H", lambda: ev.evaluate_hessian_values(x0, 1.0, lam)),
    ):
        samples = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t)
        per_call[label] = statistics.median(samples)

    before = _timing.snapshot()
    t0 = time.perf_counter()
    result = solve_nlp(ev, x0, options={"print_level": 0, "max_iter": 200})
    wall = time.perf_counter() - t0
    buckets = _timing.since(before)

    counts = {}
    info_evals = getattr(result, "raw_info", None)
    print(
        f"  solve: {result.status.name} in {wall:.3f} s, {result.iterations} iterations", flush=True
    )
    print(
        f"  _timing buckets: { ({k: round(v, 4) for k, v in buckets.items() if v > 0}) }",
        flush=True,
    )

    # Counter-based estimate. POUNCE reports n_obj/constr/grad/jac/hess evals;
    # solve_nlp does not surface the raw info dict, so re-derive from iterations
    # when it is unavailable and SAY SO rather than quietly reporting one number.
    est = None
    if info_evals is None:
        iters = max(int(result.iterations), 1)
        est = iters * (per_call["g"] + per_call["grad"] + per_call["J"] + per_call["H"])
        counts = {"source": "iterations x one call of each (upper-bound proxy)", "iters": iters}
    print(
        f"  per-call: g {per_call['g'] * 1e3:.3f} ms  grad {per_call['grad'] * 1e3:.3f} ms  "
        f"J {per_call['J'] * 1e3:.3f} ms  H {per_call['H'] * 1e3:.3f} ms",
        flush=True,
    )
    if est is not None:
        print(
            f"  evaluation share (counter estimate, {counts['source']}): "
            f"{est:.3f} s of {wall:.3f} s = {100 * est / wall:.1f}%",
            flush=True,
        )
    rust = buckets.get("rust", 0.0)
    if rust > 0:
        print(
            f"  evaluation share (_timing 'rust' bucket): {rust:.3f} s of {wall:.3f} s = "
            f"{100 * rust / wall:.1f}%",
            flush=True,
        )
    else:
        print("  _timing 'rust' bucket is ZERO — that instrument measured nothing here", flush=True)
    return {
        "K": K,
        "n": int(n),
        "m": int(m),
        "wall_s": wall,
        "iterations": int(result.iterations),
        "status": result.status.name,
        "per_call_s": per_call,
        "buckets": buckets,
        "counter_estimate_s": est,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="8,32")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    runs = [
        measure(K, args.steps, args.dim, args.reps) for K in (int(t) for t in args.ks.split(","))
    ]
    if not runs:
        print("NO MEASUREMENTS TAKEN", flush=True)
        return 1
    print(f"\nmeasurements taken: {len(runs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
