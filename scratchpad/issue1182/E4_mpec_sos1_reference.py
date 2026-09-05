"""#1182 entry experiment, part 5: the SOS1 reference requirement 4 names.

Requirement 4 of #1182 asks for a benchmark "against POUNCE and exact GDP/SOS1
references, recording source feasibility and solve guarantees separately from
time". E1/E2 cover the exact **GDP** references (big-M, hull) on general
disjunctions. SOS1 in this tree is not a general disjunction lowering: it is one
of ``discopt.mpec``'s complementarity encodings, so the SOS1 comparison only
exists on MPEC models. That is what this probe measures.

A complementarity ``0 <= f _|_ g >= 0`` is ``f >= 0, g >= 0, (f == 0 or g == 0)``,
which ``reformulate_gdp`` in ``discopt.mpec`` states as an ``either_or``. So the
same relation is reachable through four exact encodings:

* ``method="sos1"``    -- a Special Ordered Set of type 1;
* ``method="gdp"``     -- selector binary + big-M (the module default);
* ``method="gdp"`` with ``gdp_method="hull"``;
* ``method="gdp"`` with ``gdp_method="simplex"`` -- Theorem 1, no binaries at all.

All four go through the same certified ``Model.solve``. ``method="scholtes"`` is
deliberately absent: it is a homotopy of *local* NLP solves and its result is not
a certificate, so putting it in this table would compare two different kinds of
answer (the local-vs-certified distinction #1148/#1158 exists for).

Source feasibility is measured on the DECLARED operands -- ``min(f, g)`` at the
returned point -- and reported beside, never instead of, the solve guarantee and
the time.

Prints an executed-comparison count and exits non-zero if it is zero.
"""

from __future__ import annotations

import argparse
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt.mpec import complementarity, solve_mpec


def build_distance():
    """min (x-1)^2 + (y-1)^2 s.t. 0 <= x _|_ y >= 0. Optimum 1 at (1, 0) or (0, 1)."""
    m = dm.Model("mpec_distance")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    return m, [("x", "y")], [complementarity(x, y, name="c0")]


def build_chain(n: int = 4):
    """A chain of n complementarity pairs coupled by one linear row.

    Bigger than the toy so the encodings can actually separate: the optimum
    drives each pair to one side, and the coupling row makes the choice
    non-trivial.
    """
    m = dm.Model(f"mpec_chain{n}")
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n)]
    ys = [m.continuous(f"y{i}", lb=0, ub=10) for i in range(n)]
    m.minimize(sum((xs[i] - 1 - 0.3 * i) ** 2 + (ys[i] - 0.5) ** 2 for i in range(n)))
    m.subject_to(sum(xs) + sum(ys) >= 2.0)
    names = [(f"x{i}", f"y{i}") for i in range(n)]
    return m, names, [complementarity(xs[i], ys[i], name=f"c{i}") for i in range(n)]


BUILDERS = {"distance": build_distance, "chain4": lambda: build_chain(4)}

ARMS = (
    ("sos1", {}),
    ("gdp/big-m", {"gdp_method": "big-m"}),
    ("gdp/hull", {"gdp_method": "hull"}),
    ("gdp/simplex", {"gdp_method": "simplex"}),
)


def source_residual(point, pair_names) -> float:
    """max_i min(f_i, g_i) on the DECLARED operands. <= 0 means complementary."""
    worst = -np.inf
    for f_name, g_name in pair_names:
        f = float(np.asarray(point[f_name]))
        g = float(np.asarray(point[g_name]))
        worst = max(worst, min(f, g))
    return worst


def run(names, time_limit: float) -> int:
    comparisons = 0
    table = []
    for name in names:
        for arm, kwargs in ARMS:
            model, pair_names, pairs = BUILDERS[name]()
            method = "sos1" if arm == "sos1" else "gdp"
            t0 = time.perf_counter()
            try:
                res = solve_mpec(model, pairs, method=method, time_limit=time_limit, **kwargs)
            except Exception as exc:
                print(f"[{name}/{arm}] RAISED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                table.append((name, arm, "raised", None, None, None, None, None))
                comparisons += 1
                continue
            wall = time.perf_counter() - t0
            residual = source_residual(res.x, pair_names) if res.x else None
            comparisons += 1 + len(pair_names)
            table.append(
                (name, arm, str(res.status), res.objective, res.bound,
                 res.node_count, wall, residual, getattr(res, "gap_certified", None))
            )
            print(
                f"[{name}/{arm}] status={res.status} obj={res.objective} "
                f"bound={res.bound} nodes={res.node_count} wall={wall:.2f}s "
                f"certified={getattr(res, 'gap_certified', None)} "
                f"source_min(f,g)={residual}",
                flush=True,
            )

    print("\n=== summary (guarantee and source feasibility beside, not instead of, time) ===")
    hdr = ("model", "arm", "status", "objective", "bound", "nodes", "wall",
           "src min(f,g)", "certified")
    print(" | ".join(f"{h:>13}" for h in hdr))
    for row in table:
        name, arm, status, obj, bound, nodes, wall, residual = row[:8]
        certified = row[8] if len(row) > 8 else None
        print(" | ".join(f"{c:>13}" for c in (
            name, arm, status,
            "-" if obj is None else f"{obj:.6g}",
            "-" if bound is None else f"{bound:.6g}",
            "-" if nodes is None else str(nodes),
            "-" if wall is None else f"{wall:.2f}",
            "-" if residual is None else f"{residual:.3g}",
            str(certified),
        )))

    print(f"\nexecuted comparisons: {comparisons}")
    if comparisons == 0:
        print("FAIL: the probe measured nothing (CLAUDE.md section 6)")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(BUILDERS))
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()
    sys.exit(run([n for n in args.instances.split(",") if n], args.time_limit))
