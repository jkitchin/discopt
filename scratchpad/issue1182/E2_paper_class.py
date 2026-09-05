"""#1182 entry experiment, part 2: the paper's OWN class.

E1 measured the in-repo native GDP corpus and found the Theorem-1 lowering slower
and less often certified than big-M/hull. Before recording that as a
falsification it is worth asking whether the three corpus models are simply not
the class arXiv:2601.03906v1 targets: its section 5 is *optimal control with
obstacle avoidance*, where each disjunct is a single linear predicate, so the CNF
distribution is a no-op (1 clause per disjunction) and the lowering is at its
smallest and most favourable.

This probe builds exactly that: a discrete-time double integrator that must stay
outside a rectangular obstacle at every step,

    OR( x_t <= xa,  x_t >= xb,  y_t <= ya,  y_t >= yb )

which is a 4-way disjunction of single predicates -- one CNF clause, four
weights, no blowup. If the lowering loses here too, it loses on the class it was
designed for, and the falsification is not an artefact of instance selection.

Both arms are solved by the SAME certified global path (``Model.solve``), which
is the comparison #1182 says the paper does not make: its section 5 uses local
Ipopt on both sides.
"""

from __future__ import annotations

import argparse
import sys
import time

from discopt import Model
from simplex_proto import reformulate_simplex
from source_check import predicate_report

# Obstacle: the box [XA, XB] x [YA, YB], to be avoided.
XA, XB, YA, YB = 2.0, 4.0, 2.0, 4.0
START = (0.0, 0.0)
GOAL = (6.0, 6.0)


def build_avoidance(n_steps: int) -> Model:
    """Minimum-effort double integrator from START to GOAL avoiding the box."""
    m = Model(f"avoid{n_steps}")
    x = m.continuous("x", n_steps + 1, lb=-1.0, ub=8.0)
    y = m.continuous("y", n_steps + 1, lb=-1.0, ub=8.0)
    vx = m.continuous("vx", n_steps + 1, lb=-5.0, ub=5.0)
    vy = m.continuous("vy", n_steps + 1, lb=-5.0, ub=5.0)
    ax = m.continuous("ax", n_steps, lb=-3.0, ub=3.0)
    ay = m.continuous("ay", n_steps, lb=-3.0, ub=3.0)
    dt = 1.0

    m.subject_to(x[0] == START[0])
    m.subject_to(y[0] == START[1])
    m.subject_to(vx[0] == 0.0)
    m.subject_to(vy[0] == 0.0)
    m.subject_to(x[n_steps] == GOAL[0])
    m.subject_to(y[n_steps] == GOAL[1])

    for t in range(n_steps):
        m.subject_to(x[t + 1] == x[t] + dt * vx[t])
        m.subject_to(y[t + 1] == y[t] + dt * vy[t])
        m.subject_to(vx[t + 1] == vx[t] + dt * ax[t])
        m.subject_to(vy[t + 1] == vy[t] + dt * ay[t])

    # Obstacle avoidance at every step: one 4-way disjunction of single predicates.
    for t in range(n_steps + 1):
        m.either_or(
            [
                [x[t] <= XA],
                [x[t] >= XB],
                [y[t] <= YA],
                [y[t] >= YB],
            ],
            name=f"avoid_{t}",
        )

    m.minimize(sum(ax[t] * ax[t] + ay[t] * ay[t] for t in range(n_steps)))
    return m


def _nnz(model) -> int:
    from discopt.modeling.core import Constraint
    from source_check import variables_in

    return sum(len(variables_in(c.body)) for c in model._constraints if isinstance(c, Constraint))


def run(steps: list[int], time_limit: float, reps: int) -> int:
    comparisons = 0
    table = []
    # Interleave arms within a repetition (CLAUDE.md section 9) rather than
    # running one arm to completion and then the other.
    for rep in range(reps):
        for n in steps:
            for arm in ("big-m", "hull", "simplex"):
                source = build_avoidance(n)
                if arm == "simplex":
                    target, _records, counts = reformulate_simplex(build_avoidance(n))
                    counts.jacobian_nonzeros = _nnz(target)
                    t0 = time.perf_counter()
                    res = target.solve(time_limit=time_limit)
                    wall = time.perf_counter() - t0
                else:
                    from simplex_proto import SizeCounts

                    counts = SizeCounts()
                    t0 = time.perf_counter()
                    res = build_avoidance(n).solve(time_limit=time_limit, gdp_method=arm)
                    wall = time.perf_counter() - t0

                rep_src = predicate_report(source, res.x) if res.x else None
                comparisons += 1 + (0 if rep_src is None else rep_src.comparisons)
                table.append(
                    (
                        rep, n, arm, str(res.status),
                        res.objective, res.bound, res.node_count, wall,
                        getattr(res, "gap_certified", None),
                        None if rep_src is None else rep_src.max_disjunction_violation,
                    )
                )
                print(
                    f"[rep{rep} n={n} {arm}] status={res.status} obj={res.objective} "
                    f"bound={res.bound} nodes={res.node_count} wall={wall:.2f}s "
                    f"certified={getattr(res, 'gap_certified', None)} "
                    f"src_viol={None if rep_src is None else rep_src.max_disjunction_violation}",
                    flush=True,
                )

    print("\n=== summary ===")
    hdr = ("rep", "steps", "arm", "status", "objective", "bound", "nodes", "wall",
           "certified", "src_viol")
    print(" | ".join(f"{h:>11}" for h in hdr))
    for row in table:
        cells = [
            str(row[0]), str(row[1]), row[2], row[3],
            "-" if row[4] is None else f"{row[4]:.6g}",
            "-" if row[5] is None else f"{row[5]:.6g}",
            str(row[6]), f"{row[7]:.2f}", str(row[8]),
            "-" if row[9] is None else f"{row[9]:.3g}",
        ]
        print(" | ".join(f"{c:>11}" for c in cells))

    print(f"\nexecuted comparisons: {comparisons}")
    if comparisons == 0:
        print("FAIL: the probe measured nothing (CLAUDE.md section 6)")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="3,5,8")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    sys.exit(run([int(s) for s in args.steps.split(",")], args.time_limit, args.reps))
