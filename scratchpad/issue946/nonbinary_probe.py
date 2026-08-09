"""Issue #946 item 3: what happens with a *non-binary* first stage and the same
degenerate recourse? The integer L-shaped cut needs 0/1 flips, so it is not
available; the honest outcome is an uncertified bound, and GBD should say why
instead of silently exhausting its iteration budget.

Prints an executed-measurement count and exits non-zero if nothing was measured.
"""

from __future__ import annotations

import logging
import sys

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
from discopt.decomposition.benders import solve_benders

CHECKS = 0


def run(model_fn, label: str, bound_relax: float | None):
    global CHECKS
    real = nlp_pounce.solve_nlp
    calls = {"n": 0}

    def patched(problem, x0, options=None):
        opts = dict(options or {})
        if bound_relax is not None:
            opts["bound_relax_factor"] = bound_relax
        calls["n"] += 1
        return real(problem, x0, options=opts)

    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    gbd_logger = logging.getLogger("discopt.decomposition.benders.gbd")
    handler = Collect()
    gbd_logger.addHandler(handler)
    gbd_logger.setLevel(logging.WARNING)
    nlp_pounce.solve_nlp = patched  # type: ignore[assignment]
    try:
        res = solve_benders(model_fn(), time_limit=60)
    finally:
        nlp_pounce.solve_nlp = real  # type: ignore[assignment]
        gbd_logger.removeHandler(handler)

    print(f"\n=== {label} (bound_relax={bound_relax}) ===", flush=True)
    print(f"  status={res.status} objective={res.objective!r} bound={res.bound!r}")
    print(f"  recourse NLP solves: {calls['n']}")
    for r in records:
        print(f"  WARNING: {r.getMessage()}")
    CHECKS += 1
    return res, records


def integer_master():
    """Same degenerate recourse, but a 0..3 *integer* first stage (not 0/1)."""
    m = dm.Model("linnl_int")
    y = m.integer("y", lb=0, ub=3)
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


def binary_master():
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


def main() -> int:
    run(binary_master, "binary master", 0.0)
    run(integer_master, "integer master 0..3", None)
    run(integer_master, "integer master 0..3", 0.0)
    print(f"\nexecuted measurements: {CHECKS}")
    if CHECKS == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
