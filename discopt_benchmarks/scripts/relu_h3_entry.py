"""H3 entry experiment for #2167: can Anderson cuts tighten the native relaxation?

H3 (issue #2167): continuous-``q`` Anderson cuts tighten the native
complementarity relaxation beyond the triangle. Kill criterion: drop the option
if it closes less than 10% of the remaining root gap on median.

The experiment needs no implementation of the option, because both ends of the
comparison already exist as LP relaxations:

* the native root relaxation of a pair is the triangle ``y/u + s/(-l) <= 1``,
  which is exactly ``relu_bigm``'s LP relaxation with the same interval bounds
  (the identity stated in the issue; checked here, not assumed: the script also
  solves the native model's own root and compares);
* every Anderson inequality is valid for the per-neuron convex hull, so the most
  any set of them can reach is ``relu_extended``'s LP relaxation (the hull).

The option was built and measured directly as well (bound after 100 nodes,
cuts vs none, same tree), then RETIRED with its implementation: median 0.0%
of the gap closed over 16 instances, +25% wall
(docs/dev/issue-2167-relu-complementarity-results.md,
docs/dev/data/2167/h3_direct.jsonl). This script keeps the LP proxy.

So ``(LP_ext - LP_bigm) / (opt - LP_bigm)`` is an UPPER bound on the root gap
the option could close. If it is below 10% on median, the option fails its kill
criterion without being built.

Each line is one JSON record; the run ends with the number of comparisons made
and exits non-zero when it is zero (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import discopt.modeling as dm
import numpy as np
from discopt.ml.formulations import NNFormulation
from discopt.modeling.core import Model
from discopt.transformations import get as get_transformation
from discopt_benchmarks.scripts.relu_local_h1 import make_net

PANEL = [(4, 8, 2), (4, 16, 2), (4, 32, 2), (4, 16, 3), (8, 16, 2), (3, 24, 2), (2, 32, 2)]


def lp_bound(net, strategy: str) -> float:
    """Optimal value of the strategy's LP relaxation (integrality dropped)."""
    m = Model(f"lp_{strategy}")
    nn = NNFormulation(m, net, strategy=strategy)
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    get_transformation("core.relax_integrality").apply(m)
    r = m.solve()
    assert r.status == "optimal" and r.gap_certified, (strategy, r.status)
    return float(r.objective)


def milp_opt(net, time_limit: float) -> tuple[float, bool]:
    m = Model("opt")
    nn = NNFormulation(m, net, strategy="relu_bigm")
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    r = m.solve(time_limit=time_limit)
    return float(r.objective), bool(r.gap_certified)


def native_root(net) -> float:
    """Root bound of the natively lowered model (one node)."""
    from discopt.mpec import solve_mpec

    m = Model("native")
    nn = NNFormulation(m, net, strategy="relu_complementarity")
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    r = solve_mpec(m, nn.relations, method="native", max_nodes=1)
    return float(r.root_bound) if r.root_bound is not None else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--native-root", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    n_cmp = 0
    closed = []
    for n_in, width, depth in PANEL:
        for seed in [int(s) for s in args.seeds.split(",")]:
            net = make_net(seed, n_in, width, depth)
            opt, cert = milp_opt(net, args.time_limit)
            if not cert:
                print(
                    json.dumps(
                        {
                            "n_in": n_in,
                            "width": width,
                            "depth": depth,
                            "seed": seed,
                            "skipped": "optimum not certified",
                        }
                    ),
                    flush=True,
                )
                continue
            lb_bigm = lp_bound(net, "relu_bigm")
            lb_ext = lp_bound(net, "relu_extended")
            rec = {
                "n_in": n_in,
                "width": width,
                "depth": depth,
                "seed": seed,
                "opt": opt,
                "lp_bigm": lb_bigm,
                "lp_extended": lb_ext,
            }
            assert lb_bigm <= lb_ext + 1e-7 <= opt + 2e-7, rec
            gap = opt - lb_bigm
            rec["closed"] = None if gap <= 1e-9 else (lb_ext - lb_bigm) / gap
            if args.native_root:
                rec["native_root"] = native_root(net)
            if rec["closed"] is not None:
                closed.append(rec["closed"])
            n_cmp += 1
            print(json.dumps(rec), flush=True)
    if closed:
        print(
            f"# median gap closed by the hull (upper bound for H3): {np.median(closed):.4f} "
            f"over {len(closed)} instances; kill if < 0.10",
            flush=True,
        )
    print(f"# comparisons made: {n_cmp}", flush=True)
    return 0 if n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
