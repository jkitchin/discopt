"""#1370 Part B: measure the SHIPPED path, not the harness.

The entry experiment hand-wrote the compressed arm. This measures
`discopt._block_eval.CompressedBlockEvaluator` — the implementation — against
the default tape evaluator, on the derivatives and end to end through a solve.

Two things this corrects about the harness, both in the conservative direction:

  * the harness coloured the Hessian with a distance-1 coloring of the Hessian
    graph, which is **not** sufficient for direct recovery (two columns of the
    same colour can share a neighbouring row). The implementation uses
    `_relax/sparse_hessian.build_hessian_coloring`, which is correct and needs
    more seeds — 21 rather than 1 on the K=8 model.
  * the harness's model has a diagonal Lagrangian Hessian (the `sin` is
    elementwise), which flatters any Hessian compression. `--coupled` adds a
    neighbour product to the dynamics so the Hessian is genuinely banded.

Run: python -u scratchpad/issue1370/measure_impl.py --ks 8,32 --coupled
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import Model

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from block_eval_entry import load_gate, make_block_model  # noqa: E402

H_STEP = 0.1


def make_coupled_block_model(K: int, steps: int, dim: int) -> Model:
    """Same class, but with a neighbour product so the Hessian is not diagonal.

    `z[k,t,i] * z[k,t+1,i]` in the dynamics puts off-diagonal entries in the
    Lagrangian Hessian, which is what a real collocation/AC-power block has and
    what a compression measurement has to be taken against.
    """
    m = Model(f"coupled_K{K}")
    w = m.continuous("w", shape=(dim,), lb=-2.0, ub=2.0)
    z = m.continuous("z", shape=(K, steps, dim), lb=-10.0, ub=10.0)
    zc = z[:, :-1, :]
    # Couple across TIME, elementwise: z[t] * z[t+1] puts off-diagonal entries in
    # the Lagrangian Hessian. (Across the last axis it would need a shift matmul,
    # which the tape does not lower for stacked operands, or a concatenate, which
    # returns an object array the modeling layer refuses to multiply elementwise.)
    rhs = -w[None, None, :] * zc + 0.1 * dm.sin(zc) + 0.05 * zc * z[:, 1:, :]
    m.subject_to(z[:, 1:, :] - zc - H_STEP * rhs == 0.0, name="dyn")
    m.minimize(dm.sum((z - 1.0) ** 2) + 0.01 * dm.sum(w**2))
    m.set_block(z, np.repeat(np.arange(K, dtype=np.int64), steps * dim).reshape(K, steps, dim))
    m.set_block(w, -1)
    return m


def run_one(K: int, steps: int, dim: int, reps: int, coupled: bool) -> dict:
    from discopt._block_eval import build_compressed_evaluator
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    model = make_coupled_block_model(K, steps, dim) if coupled else make_block_model(K, steps, dim)
    base = TapeNLPEvaluator(model)
    n, m = base.n_variables, base.n_constraints
    print(f"\n=== K={K} steps={steps} dim={dim} coupled={coupled}: n={n} m={m} ===", flush=True)

    t0 = time.perf_counter()
    comp = build_compressed_evaluator(model, base)
    print(f"  {comp.summary()}, built in {time.perf_counter() - t0:.2f}s", flush=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    lam = rng.normal(size=m)

    # The builder already checked agreement; check again at THIS point, and
    # count, so a run that compared nothing cannot report a speedup.
    bj, cj = base.evaluate_jacobian_values(x), comp.evaluate_jacobian_values(x)
    bh, ch = base.evaluate_hessian_values(x, 1.0, lam), comp.evaluate_hessian_values(x, 1.0, lam)
    compared = int(bj.size + bh.size)
    diff = max(float(np.max(np.abs(bj - cj))), float(np.max(np.abs(bh - ch))))
    print(f"  agreement: {compared} entries, max|diff| {diff:.3e}", flush=True)
    if compared == 0 or diff > 1e-9:
        raise SystemExit("disagreement or empty comparison; the timings would be meaningless")

    out: dict = {
        "K": K,
        "n": int(n),
        "m": int(m),
        "colors": comp.n_jacobian_colors,
        "hess_seeds": comp.n_hessian_seeds,
        "max_abs_diff": diff,
    }
    for label, a, b in (
        (
            "jacobian",
            lambda: base.evaluate_jacobian_values(x),
            lambda: comp.evaluate_jacobian_values(x),
        ),
        (
            "hessian",
            lambda: base.evaluate_hessian_values(x, 1.0, lam),
            lambda: comp.evaluate_hessian_values(x, 1.0, lam),
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
        out[label] = {
            "tape_s": am,
            "tape_sd": asd,
            "compressed_s": bm,
            "compressed_sd": bsd,
            "speedup": am / bm,
        }
        print(
            f"  {label:9s} tape {am * 1e3:9.3f} ms (sd {asd * 1e3:6.3f})  "
            f"compressed {bm * 1e3:9.3f} ms (sd {bsd * 1e3:6.3f})  speedup {am / bm:6.2f}x",
            flush=True,
        )

    # End to end, interleaved: the number that actually matters.
    from discopt.solvers.nlp_pounce import solve_nlp

    x0 = rng.normal(size=n) * 0.1
    opts = {"print_level": 0, "max_iter": 200}
    walls: dict = {"tape": [], "compressed": []}
    statuses = set()
    objs = []
    for _ in range(max(3, reps // 2)):
        for label, ev in (("tape", base), ("compressed", comp)):
            t = time.perf_counter()
            r = solve_nlp(ev, x0, options=opts)
            walls[label].append(time.perf_counter() - t)
            statuses.add(r.status.name)
            objs.append(r.objective)
    tw, cw = statistics.median(walls["tape"]), statistics.median(walls["compressed"])
    out["solve"] = {
        "tape_s": tw,
        "compressed_s": cw,
        "speedup": tw / cw,
        "statuses": sorted(statuses),
        "objective_spread": max(objs) - min(objs),
    }
    print(
        f"  SOLVE     tape {tw:7.3f} s   compressed {cw:7.3f} s   speedup {tw / cw:5.2f}x   "
        f"statuses {sorted(statuses)}   objective spread {max(objs) - min(objs):.3e}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="8,32")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--coupled", action="store_true")
    ap.add_argument("--allow-load", action="store_true")
    args = ap.parse_args()

    print("#1370 Part B: the shipped compressed evaluator vs the default tape")
    load = load_gate()
    if load > 2.0 and not args.allow_load:
        print(f"REFUSING: load {load:.2f} > 2.0 (CLAUDE.md §9).", flush=True)
        return 2

    runs = [
        run_one(K, args.steps, args.dim, args.reps, args.coupled)
        for K in (int(t) for t in args.ks.split(","))
    ]
    print(f"\nmeasurements taken: {len(runs)}", flush=True)
    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
