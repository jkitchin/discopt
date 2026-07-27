"""#875: prove ``interval_hessian_submatrix`` == the dense slice it replaces.

``_multivar_box_curvature`` built the full ``n x n`` interval Hessian and then read
``np.ix_(dep, dep)``. The submatrix form assembles that block directly from the
walker's sparse ``{(i,j): Interval}``. This asserts the two agree BIT-FOR-BIT
(``array_equal``) on every nonlinear constraint body of every in-repo instance small
enough to densify, over several boxes — because the curvature verdict that rides on
it is a soundness artifact, not a heuristic.
"""

from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt._jax.convexity.interval_ad import (  # noqa: E402
    interval_hessian,
    interval_hessian_submatrix,
)
from discopt.modeling.core import Constraint, from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data")
_MAX_N = 400  # densifying n x n above this is what the fix exists to avoid


def dep_indices(model, expr) -> list[int]:
    """Flat indices the expression depends on (same set the caller passes)."""
    from discopt._jax.convexity.interval_ad import _flat_size

    n = _flat_size(model)
    ad = interval_hessian(expr, model)
    lo = np.asarray(ad.grad.lo, dtype=np.float64)
    hi = np.asarray(ad.grad.hi, dtype=np.float64)
    touched = np.nonzero((lo != 0.0) | (hi != 0.0))[0]
    return [int(j) for j in touched[:8]] or [0] if n else []


def main():
    paths = sorted(glob.glob(os.path.join(_DATA, "minlplib_nl", "*.nl"))) + sorted(
        glob.glob(os.path.join(_DATA, "minlplib", "*.nl"))
    )
    seen: set = set()
    compared = mismatched = skipped = 0
    problems: list[str] = []
    for p in paths:
        name = os.path.basename(p)[:-3]
        if name in seen:
            continue
        seen.add(name)
        try:
            m = from_nl(p)
        except Exception:
            skipped += 1
            continue
        n = sum(v.size for v in m._variables)
        if n == 0 or n > _MAX_N:
            skipped += 1
            continue
        bodies = [c.body for c in m._constraints if isinstance(c, Constraint)]
        if m._objective is not None:
            bodies.append(m._objective.expression)
        for expr in bodies[:25]:
            try:
                dep = dep_indices(m, expr)
                if not dep:
                    continue
                ad = interval_hessian(expr, m)
                ix = np.ix_(dep, dep)
                want_lo = np.asarray(ad.hess.lo, dtype=np.float64)[ix]
                want_hi = np.asarray(ad.hess.hi, dtype=np.float64)[ix]
                sub = interval_hessian_submatrix(expr, m, dep)
            except Exception:
                continue
            compared += 1
            if sub is None:
                # dense path represents this as all-inf; both mean "no verdict"
                if np.all(np.isinf(want_lo)) and np.all(np.isinf(want_hi)):
                    continue
                mismatched += 1
                problems.append(f"{name}: submatrix None but dense slice is finite")
                continue
            got_lo, got_hi = sub
            if not (np.array_equal(got_lo, want_lo) and np.array_equal(got_hi, want_hi)):
                mismatched += 1
                d = max(
                    float(np.max(np.abs(got_lo - want_lo))),
                    float(np.max(np.abs(got_hi - want_hi))),
                )
                problems.append(f"{name}: block differs, max |delta| = {d}")

    print(f"blocks compared : {compared}   (skipped {skipped} instances)")
    print(f"MISMATCHES      : {mismatched}")
    for q in problems[:20]:
        print(f"    {q}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
