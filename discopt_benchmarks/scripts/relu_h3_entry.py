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


def bound_after_nodes(net, cuts: bool, nodes: int) -> tuple[float, int, float]:
    """Global dual bound of the native model after ``nodes`` nodes (direct H3).

    Not the root bound: callback cuts reach the relaxation only after the root
    LP is solved, so ``root_bound`` is identical with and without them (measured:
    4 instances, 4-19 cuts each, bounds equal to 4 digits). Both arms run on the
    Python tree (``DISCOPT_COMPL_BRANCH=1``; any ``cut_callback`` already bypasses
    the native kernel), so the only difference is the q columns and their cuts.
    """
    import os

    from discopt.mpec import solve_mpec

    assert os.environ.get("DISCOPT_COMPL_BRANCH") == "1", "run both arms on one tree"
    m = Model("native_cuts")
    nn = NNFormulation(m, net, strategy="relu_complementarity", ideal_cuts=cuts)
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    kw = {"cut_callback": nn.cut_callback} if cuts else {}
    r = solve_mpec(m, nn.relations, method="native", max_nodes=nodes, deterministic=True, **kw)
    n_cuts = nn.cut_callback.n_cuts if cuts else 0
    return (float(r.bound) if r.bound is not None else float("nan")), n_cuts, r.wall_time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--native-root", action="store_true")
    ap.add_argument(
        "--direct", type=int, default=0, help="node budget for the direct cuts-vs-plain arm"
    )
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    n_cmp = 0
    closed = []
    direct: list[float] = []
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
            if args.direct:
                r0, _, w0 = bound_after_nodes(net, False, args.direct)
                r1, nc, w1 = bound_after_nodes(net, True, args.direct)
                rec.update(bound_plain=r0, bound_cuts=r1, n_cuts=nc, wall_plain=w0, wall_cuts=w1)
                assert r1 <= opt + 1e-6 and r0 <= opt + 1e-6, rec  # no bound above opt
                g = opt - r0
                rec["closed_direct"] = None if g <= 1e-6 * max(1.0, abs(opt)) else (r1 - r0) / g
                if rec["closed_direct"] is not None:
                    direct.append(rec["closed_direct"])
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
    if direct:
        print(
            f"# DIRECT median root gap closed by the cuts: {np.median(direct):.4f} over "
            f"{len(direct)} instances; kill if < 0.10",
            flush=True,
        )
    print(f"# comparisons made: {n_cmp}", flush=True)
    return 0 if n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
