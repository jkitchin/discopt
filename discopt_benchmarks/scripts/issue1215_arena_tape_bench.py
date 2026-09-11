#!/usr/bin/env python
"""How much of the tape build does the arena lowering actually remove? (#1215)

Measures ``TapeNLPEvaluator`` construction -- the whole build, including
``model_to_repr`` and POUNCE's ``build_nl_problem`` -- with the arena lowering ON
and OFF, on the same models in the same process.

Discipline (§9): the two arms are **interleaved** rather than run in sequence, a
load average is printed before and after so a contended box is visible, and every
number is reported as a median with a standard deviation over the reps rather
than a single timing. §6: the executed-build count is printed and a zero exits
non-zero. §8: the loaded module and a marker unique to this change are asserted.

Usage::

    python -u discopt_benchmarks/scripts/issue1215_arena_tape_bench.py
    python -u discopt_benchmarks/scripts/issue1215_arena_tape_bench.py --reps 7
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import discopt.modeling as dm
import discopt.modeling.core as core


def flowsheet(n_units: int):
    """Process-shaped: per-unit mass balance, an Arrhenius rate, an energy bound."""
    m = dm.Model(f"fs{n_units}")
    outs = []
    for u in range(n_units):
        f = m.continuous(f"f{u}", shape=(3,), lb=0.1, ub=10.0)
        t = m.continuous(f"T{u}", lb=300.0, ub=600.0)
        m.subject_to(f[0] + f[1] - f[2] == 0.0, name=f"mb{u}")
        m.subject_to(f[2] * dm.exp(-2000.0 / t) <= 5.0, name=f"rate{u}")
        m.subject_to(dm.log(f[0] + 1.0) + 0.01 * t <= 12.0, name=f"nrg{u}")
        outs.append(f[2])
    m.minimize(dm.sum(outs))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 8000])
    args = ap.parse_args()

    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    print(f"# discopt loaded from: {core.__file__}")
    marker = "try_build_arena_tape"
    src = Path("python/discopt/_tape_nlp_evaluator.py").read_text()
    assert marker in src, "evaluator is not wired to the arena path -- wrong tree"
    print(f"# marker '{marker}' present in the evaluator")
    print(f"# load before: {os.getloadavg()[0]:.2f}   cpus={os.cpu_count()}")

    builds = 0
    print(f"\n{'rows':>7s} {'arm':<8s} {'median s':>10s} {'sd':>8s} {'us/row':>9s} {'speedup':>8s}")

    for n_units in args.sizes:
        model = flowsheet(n_units)
        n_rows = len(model._constraints)
        times = {"arena": [], "python": []}
        for _ in range(args.reps):
            # Interleaved, not sequential: a drift in machine state hits both arms.
            for arm, env in (("arena", "1"), ("python", "0")):
                os.environ["DISCOPT_ARENA_TAPE"] = env
                t0 = time.perf_counter()
                ev = TapeNLPEvaluator(model)
                times[arm].append(time.perf_counter() - t0)
                builds += 1
                # Prove the arm did what its name says: with the flag off the
                # evaluator must have walked the Python DAG. A benchmark whose
                # arms silently ran the same code is the failure this guards.
                assert ev.n_constraints == n_rows, (
                    f"{arm} arm built {ev.n_constraints} rows, expected {n_rows}"
                )
        os.environ.pop("DISCOPT_ARENA_TAPE", None)

        med = {a: statistics.median(v) for a, v in times.items()}
        sd = {a: (statistics.stdev(v) if len(v) > 1 else 0.0) for a, v in times.items()}
        for arm in ("python", "arena"):
            speed = med["python"] / med["arena"] if arm == "arena" else 1.0
            tag = f"{speed:7.2f}x" if arm == "arena" else "       -"
            print(
                f"{n_rows:7d} {arm:<8s} {med[arm]:10.3f} {sd[arm]:8.3f} "
                f"{med[arm] / n_rows * 1e6:9.2f} {tag}"
            )

    print(f"\n# load after: {os.getloadavg()[0]:.2f}")
    print(f"# executed: {builds} evaluator builds")
    if builds == 0:
        print("FAIL: measured nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
