"""#1370 Part B, third measurement: is the vectorised pass already in the tree?

Before building a block-vectorised evaluator, check what `_relax/` already has.
It turns out to have the whole mechanism: `sparsity.compute_coloring` +
`make_seed_matrix`, `sparse_jacobian.make_sparse_jac_fn` (O(chromatic number)
JVPs) and `sparse_hessian.make_sparse_hess_values_fn` (colored HVPs). The JAX
evaluator routes to them behind its own gates.

That is the same mechanism the entry experiment hand-wrote — one pattern, one
coloring, one vectorised pass — so the question is not "can it be built" but
"does the thing already in the tree beat the default tape on block-structured
models, and does it agree with it".

Arms, interleaved, on identical models and points:

  * `TapeNLPEvaluator`  — today's default (POUNCE's Rust AD tape).
  * `NLPEvaluator`      — the JAX arm, on its sparse coloring path.

Values are compared before any timing is reported: an arm that disagrees is not
an alternative. Run: python -u scratchpad/issue1370/existing_paths.py --ks 8,32
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from block_eval_entry import load_gate, make_block_model  # noqa: E402


def run_one(K: int, steps: int, dim: int, reps: int) -> dict:
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    print(f"\n=== K={K} steps={steps} dim={dim} ===", flush=True)
    model = make_block_model(K, steps, dim)
    tape = TapeNLPEvaluator(model)
    n, m = tape.n_variables, tape.n_constraints
    print(f"  n={n} m={m}", flush=True)

    t0 = time.perf_counter()
    jaxev = NLPEvaluator(model)
    print(f"  JAX evaluator built in {time.perf_counter() - t0:.2f}s", flush=True)

    pattern = jaxev.sparsity_pattern
    print(
        f"  sparse structure: {jaxev.has_sparse_structure()}, "
        f"jac nnz {None if pattern is None else pattern.jacobian_nnz}, "
        f"hess density {None if pattern is None else round(pattern.hessian_density, 6)}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    lam = rng.normal(size=m)

    # Structures must agree before values can be compared at all.
    tr, tc = tape.jacobian_structure()
    jr, jc = jaxev.jacobian_structure()
    same_jac_struct = np.array_equal(np.asarray(tr), np.asarray(jr)) and np.array_equal(
        np.asarray(tc), np.asarray(jc)
    )
    thr, thc = tape.hessian_structure()
    jhr, jhc = jaxev.hessian_structure()
    same_hess_struct = np.array_equal(np.asarray(thr), np.asarray(jhr)) and np.array_equal(
        np.asarray(thc), np.asarray(jhc)
    )
    print(f"  structures identical: jac {same_jac_struct}, hess {same_hess_struct}", flush=True)

    tape_J = tape.evaluate_jacobian_values(x)
    jax_J = jaxev.evaluate_jacobian_values(x)
    tape_H = tape.evaluate_hessian_values(x, 1.0, lam)
    jax_H = jaxev.evaluate_hessian_values(x, 1.0, lam)
    compared = int(min(tape_J.size, jax_J.size) + min(tape_H.size, jax_H.size))
    if compared == 0:
        raise SystemExit("compared NOTHING; the timings below would be meaningless")
    if tape_J.shape != jax_J.shape or tape_H.shape != jax_H.shape:
        raise SystemExit(
            f"value arrays differ in shape: J {tape_J.shape} vs {jax_J.shape}, "
            f"H {tape_H.shape} vs {jax_H.shape}"
        )
    dj = float(np.max(np.abs(tape_J - jax_J)))
    dh = float(np.max(np.abs(tape_H - jax_H)))
    print(f"  values: {compared} entries compared, max|dJ| {dj:.3e}, max|dH| {dh:.3e}", flush=True)
    if max(dj, dh) > 1e-9:
        raise SystemExit("the two arms disagree; not an alternative")

    out: dict = {"K": K, "n": int(n), "m": int(m), "max_abs_diff": max(dj, dh)}
    for label, a, b in (
        (
            "jacobian",
            lambda: tape.evaluate_jacobian_values(x),
            lambda: jaxev.evaluate_jacobian_values(x),
        ),
        (
            "hessian",
            lambda: tape.evaluate_hessian_values(x, 1.0, lam),
            lambda: jaxev.evaluate_hessian_values(x, 1.0, lam),
        ),
    ):
        a()
        b()
        asm, bsm = [], []
        for _ in range(reps):
            t = time.perf_counter()
            a()
            asm.append(time.perf_counter() - t)
            t = time.perf_counter()
            b()
            bsm.append(time.perf_counter() - t)
        am, bm = statistics.median(asm), statistics.median(bsm)
        asd = statistics.stdev(asm) if reps > 1 else 0.0
        bsd = statistics.stdev(bsm) if reps > 1 else 0.0
        out[label] = {"tape_s": am, "tape_sd": asd, "jax_s": bm, "jax_sd": bsd, "speedup": am / bm}
        print(
            f"  {label:9s} tape {am * 1e3:9.3f} ms (sd {asd * 1e3:6.3f})  "
            f"jax-sparse {bm * 1e3:9.3f} ms (sd {bsd * 1e3:6.3f})  speedup {am / bm:6.2f}x",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="8,32")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--allow-load", action="store_true")
    args = ap.parse_args()

    print("#1370 Part B: the tape vs the JAX evaluator's existing sparse coloring path")
    load = load_gate()
    if load > 2.0 and not args.allow_load:
        print(f"REFUSING: load {load:.2f} > 2.0 (CLAUDE.md §9).", flush=True)
        return 2

    runs = [
        run_one(K, args.steps, args.dim, args.reps) for K in (int(t) for t in args.ks.split(","))
    ]
    print(f"\nmeasurements taken: {len(runs)}", flush=True)
    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
