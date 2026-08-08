"""Issue #946 entry experiment: does the degenerate recourse multiplier actually
cost GBD its certificate on this build?

Arms (in-process, interleaved by construction — same model object recipe, same
process, only the recourse NLP's ``bound_relax_factor`` differs):

  A  default          — Ipopt/POUNCE default bound_relax_factor (1e-8)
  B  bound_relax=0    — the arm issue #946 says diverges

Taps every ``solve_nlp`` return inside ``gbd._attempt_recourse`` so the recourse
point, the multiplier and the resulting cut data are *measured*, not re-derived.

Prints an executed-assertion count and exits non-zero if nothing was measured
(CLAUDE.md §6: prove the probe fired).
"""

from __future__ import annotations

import sys

import numpy as np

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
from discopt.decomposition.benders import solve_benders

MARKER_CHECKS = 0


def model():
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


def run(bound_relax: float | None):
    """Solve with the recourse NLP's bound_relax_factor forced to *bound_relax*.

    ``None`` leaves the backend default. Returns (result, taps)."""
    real = nlp_pounce.solve_nlp
    taps: list[dict] = []

    def patched(problem, x0, options=None):
        opts = dict(options or {})
        if bound_relax is not None:
            opts["bound_relax_factor"] = bound_relax
        res = real(problem, x0, options=opts)
        taps.append(
            {
                "x": None if res.x is None else np.asarray(res.x, dtype=float).copy(),
                "mu": None
                if res.multipliers is None
                else np.asarray(res.multipliers, dtype=float).copy(),
                "status": res.status,
            }
        )
        return res

    nlp_pounce.solve_nlp = patched  # type: ignore[assignment]
    try:
        res = solve_benders(model(), time_limit=30)
    finally:
        nlp_pounce.solve_nlp = real  # type: ignore[assignment]
    return res, taps


def report(label: str, res, taps) -> None:
    global MARKER_CHECKS
    print(f"\n=== {label} ===", flush=True)
    print(f"  status={res.status}  objective={res.objective!r}  bound={res.bound!r}")
    for k, t in enumerate(taps):
        if t["x"] is None:
            print(f"  tap[{k}]: no primal point (status={t['status']})")
            continue
        x = t["x"]
        mu = t["mu"]
        # flat order is [y, x0, x1]
        print(
            f"  tap[{k}]: y={x[0]:.6g} x=({x[1]:.6g},{x[2]:.6g}) "
            f"g=x0^2+x1^2-8y={x[1] ** 2 + x[2] ** 2 - 8 * x[0]:+.3e} "
            f"mu={'None' if mu is None else np.array2string(mu, precision=4)}"
        )
        MARKER_CHECKS += 1


def main() -> int:
    global MARKER_CHECKS
    res_a, taps_a = run(None)
    res_b, taps_b = run(0.0)
    report("A: backend default bound_relax_factor", res_a, taps_a)
    report("B: bound_relax_factor = 0", res_b, taps_b)

    print("\n--- verdict ---")
    print(f"  A: status={res_a.status} bound={res_a.bound}")
    print(f"  B: status={res_b.status} bound={res_b.bound}")
    reproduced = res_b.status != "optimal" or (
        res_b.bound is not None and res_b.bound < -1.0 - 1e-3
    )
    MARKER_CHECKS += 1
    print(f"  degeneracy reproduced on this build: {reproduced}")
    print(f"\nexecuted assertions/measurements: {MARKER_CHECKS}")
    if MARKER_CHECKS == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
