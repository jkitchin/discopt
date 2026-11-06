"""H1 panel for #2167: multistart Scholtes (``relu_complementarity``) vs ``relu_bigm``.

Hypothesis H1 (issue #2167): Scholtes on POUNCE reaches near-optimal points on
large ReLU nets much faster than ``relu_bigm`` closes them.

Kill criterion (from the issue): retire the local mode if, on nets ``relu_bigm``
closes within the budget, the median verified objective is more than 1% worse;
or if, on larger nets, it is not at least 5x faster to a point within 1% of the
best known.

Per instance, interleaved: ``relu_bigm`` with ``--bigm-time-limit``, then
``nn.solve_local`` with ``--starts`` forward-pass starts and the same wall
budget. Each line is one JSON record; the run ends with the number of
comparisons actually made and exits non-zero when it is zero (CLAUDE.md §6).

Usage::

    python -u -m discopt_benchmarks.scripts.relu_local_h1 --bigm-time-limit 120 \
        --out h1.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings

import discopt.modeling as dm
import numpy as np
from discopt.ml.formulations import NNFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.modeling.core import Model

# (n_in, width, depth, physics)
PANEL = [
    (4, 16, 2, False),
    (4, 32, 2, False),
    (4, 64, 2, False),
    (4, 128, 2, False),
    (4, 256, 2, False),
    (4, 512, 2, False),
    (4, 1024, 2, False),
    (4, 32, 3, False),
    (4, 64, 4, False),
    (4, 128, 4, False),
    (4, 64, 6, False),
    (8, 256, 3, False),
    (4, 32, 2, True),
    (4, 64, 3, True),
    (8, 128, 2, True),
]


def make_net(seed: int, n_in: int, width: int, depth: int) -> NetworkDefinition:
    """He-initialised random ReLU MLP on the box [-1, 1]^n_in, scalar output."""
    rng = np.random.default_rng(seed)
    layers, w = [], n_in
    for _ in range(depth):
        weights = rng.standard_normal((w, width)) * np.sqrt(2.0 / w)
        layers.append(DenseLayer(weights, 0.1 * rng.standard_normal(width), Activation.RELU))
        w = width
    layers.append(
        DenseLayer(rng.standard_normal((w, 1)) / np.sqrt(w), np.zeros(1), Activation.LINEAR)
    )
    return NetworkDefinition(layers, input_bounds=(-np.ones(n_in), np.ones(n_in)))


def embed(net: NetworkDefinition, strategy: str, physics: bool):
    """min output; with ``physics``, inside a small nonlinear model.

    The physics variant adds a nonconvex ring constraint on the inputs and a
    quadratic cost, so the network sits inside a genuine NLP rather than a box.
    """
    m = Model(f"h1_{strategy}")
    nn = NNFormulation(m, net, strategy=strategy)
    nn.formulate()
    x = nn.inputs
    obj = dm.sum(nn.outputs)
    if physics:
        n = net.input_size
        m.subject_to(dm.sum(x * x) >= 0.25 * n, name="ring_lo")
        m.subject_to(dm.sum(x * x) <= 0.75 * n, name="ring_hi")
        obj = obj + 0.05 * dm.sum((x - 0.3) * (x - 0.3))
    m.minimize(obj)
    return m, nn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigm-time-limit", type=float, default=120.0)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--only", type=str, default="", help="comma list of panel indices")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    import discopt.mpec as mpec

    assert hasattr(mpec, "solve_mpec_multistart"), f"wrong discopt loaded: {mpec.__file__}"
    print(f"# discopt.mpec from {mpec.__file__}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s]
    idx = [int(i) for i in args.only.split(",") if i] or list(range(len(PANEL)))
    # one JSON line per instance, flushed as written; closed at interpreter exit
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    n_cmp = 0
    for i in idx:
        n_in, width, depth, physics = PANEL[i]
        for seed in seeds:
            net = make_net(seed, n_in, width, depth)
            rec: dict = {
                "n_in": n_in,
                "width": width,
                "depth": depth,
                "physics": physics,
                "seed": seed,
            }

            mb, _ = embed(net, "relu_bigm", physics)
            t0 = time.perf_counter()
            rb = mb.solve(time_limit=args.bigm_time_limit)
            rec["bigm"] = {
                "status": str(rb.status),
                "objective": rb.objective,
                "bound": rb.bound,
                "certified": bool(rb.gap_certified),
                "wall": time.perf_counter() - t0,
            }

            mc, nn = embed(net, "relu_complementarity", physics)
            rec["n_relations"] = len(nn.relations)
            t0 = time.perf_counter()
            rl = nn.solve_local(n_starts=args.starts, seed=seed, time_limit=args.bigm_time_limit)
            wall = time.perf_counter() - t0
            setup = wall - sum(r.wall_time for r in rl.multistart)
            cum, trace = setup, []
            for r in rl.multistart:
                cum += r.wall_time
                trace.append((cum, r.verified_objective))
            rec["local"] = {
                "status": str(rl.status),
                "objective": rl.objective,
                "wall": wall,
                "starts_run": len(rl.multistart),
                "verified": sum(r.verified_objective is not None for r in rl.multistart),
                "trace": trace,
            }
            # independent check: the reported point's objective, recomputed
            if rl.x is not None:
                x_in = np.asarray(rl.x["nn_input"], dtype=np.float64)
                f = float(net.forward(x_in)[0])
                if physics:
                    f += 0.05 * float(np.sum((x_in - 0.3) ** 2))
                rec["local"]["recomputed_objective"] = f
                assert abs(f - rl.objective) <= 1e-5 * max(1.0, abs(f)), (f, rl.objective)
            n_cmp += 1
            line = json.dumps(rec)
            print(line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()
    print(f"# comparisons made: {n_cmp}", flush=True)
    return 0 if n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
