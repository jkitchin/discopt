"""Calibrate the #912 work-unit costs against measured wall time.

A deterministic work budget only behaves like the wall budget it replaces if
the unit costs are roughly proportional to real cost. This script measures, per
instance, the wall time of

  * one ``violation()`` evaluation (one constraint-vector evaluation), and
  * one continuous-repair ``subnlp`` solve,

and reports the ratio, which is what ``discopt._work_budget.NLP_SOLVE_UNITS``
encodes. It also reports the implied "units per second" rate, which is how the
default ``SolverTuning.ils_work_budget`` was sized against the legacy 5 s wall
budget.

Timing discipline (CLAUDE.md rule 9): the two measurements are **interleaved**
round by round rather than run in two blocks, the load average is printed before
and after, and every number is reported with a standard deviation over rounds.

Usage::

    python -u discopt_benchmarks/scripts/item912_work_unit_calibration.py \\
        --instances nvs21,nvs13,st_e13,ex1221 --rounds 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORPUS = os.path.join(_REPO, "python", "tests", "data", "minlplib_nl")


def _loadavg():
    return os.getloadavg()


def measure(name, rounds, evals_per_round, solves_per_round):
    import discopt
    from discopt._relax import primal_heuristics as ph
    from discopt._relax.nlp_evaluator import cached_evaluator
    from discopt.modeling.core import from_nl
    from discopt.solvers.nlp_backend import get_nlp_solver

    assert os.path.abspath(discopt.__file__).startswith(_REPO), discopt.__file__

    model = from_nl(os.path.join(_CORPUS, name + ".nl"))
    ev = cached_evaluator(model)
    lb, ub = ph._get_variable_bounds(model)
    int_mask = ph._get_integer_mask(model)
    x0 = np.clip(0.5 * (np.clip(lb, -1e3, 1e3) + np.clip(ub, -1e3, 1e3)), lb, ub)
    backend = get_nlp_solver("auto")

    # Warm the caches/JIT once so round 1 is not a cold-start outlier.
    ev.evaluate_constraints(x0)
    ph.subnlp(model, x0, backend=backend, evaluator=ev)

    eval_times, solve_times = [], []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(evals_per_round):
            ev.evaluate_constraints(x0)
        eval_times.append((time.perf_counter() - t0) / evals_per_round)

        t0 = time.perf_counter()
        for k in range(solves_per_round):
            xk = x0.copy()
            # Perturb an integer so successive solves are not identical work.
            if np.any(int_mask):
                j = int(np.where(int_mask)[0][k % int(int_mask.sum())])
                xk[j] = float(np.clip(round(xk[j]) + (k % 3) - 1, lb[j], ub[j]))
            ph.subnlp(model, xk, backend=backend, evaluator=ev)
        solve_times.append((time.perf_counter() - t0) / solves_per_round)

    return {
        "instance": name,
        "n_vars": int(x0.size),
        "n_cons": int(ev.n_constraints),
        "n_int": int(int_mask.sum()),
        "eval_s": statistics.mean(eval_times),
        "eval_sd": statistics.stdev(eval_times) if len(eval_times) > 1 else 0.0,
        "solve_s": statistics.mean(solve_times),
        "solve_sd": statistics.stdev(solve_times) if len(solve_times) > 1 else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default="nvs21,nvs13,st_e13,ex1221,nvs07,tls2")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--solves", type=int, default=20)
    args = ap.parse_args(argv)

    print(f"loadavg before: {_loadavg()}")
    rows = []
    for name in [n.strip() for n in args.instances.split(",") if n.strip()]:
        path = os.path.join(_CORPUS, name + ".nl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows.append(measure(name, args.rounds, args.evals, args.solves))
        r = rows[-1]
        print(
            f"{r['instance']:<12} vars={r['n_vars']:<5} cons={r['n_cons']:<5} "
            f"int={r['n_int']:<4} eval={r['eval_s'] * 1e6:8.1f}us "
            f"(sd {r['eval_sd'] * 1e6:6.1f}) solve={r['solve_s'] * 1e3:8.2f}ms "
            f"(sd {r['solve_sd'] * 1e3:6.2f}) ratio={r['solve_s'] / r['eval_s']:8.0f}",
            flush=True,
        )
    print(f"loadavg after:  {_loadavg()}")

    if not rows:
        print("FAIL: zero instances measured.")
        return 2
    ratios = [r["solve_s"] / r["eval_s"] for r in rows]
    print()
    print(f"measured instances:      {len(rows)}")
    print(f"solve/eval ratio geomean {np.exp(np.mean(np.log(ratios))):.0f}")
    print(f"                  min/max {min(ratios):.0f} / {max(ratios):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
