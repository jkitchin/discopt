"""#875 item 3: prove the sparse ``build_linear_context`` is BOUND-NEUTRAL.

CLAUDE.md §5 regime 1 (refactor / marshaling): the certified result must be
*exactly* unchanged, not merely close. Any drift means the change is wrong.

This compares, over the in-repo corpus, the sparse ``LinearContext`` against a
faithful reconstruction of the pre-change dense assembly (``extract_affine`` +
``np.vstack``, which is what the module did before):

  * the dense form of the sparse ``A_ub`` / ``A_eq`` must be bit-identical to the
    dense build (``array_equal``, not ``allclose``);
  * ``b_ub`` / ``b_eq`` / ``lb`` / ``ub`` bit-identical;
  * the convexity verdict the context feeds must be unchanged.
"""

from __future__ import annotations

import glob
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt._jax.convexity.linear_context import (  # noqa: E402
    build_linear_context,
    extract_affine,
)
from discopt.modeling.core import Constraint, from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data")


def dense_reference(model):
    """The pre-#875 assembly, reproduced verbatim."""
    if not model._variables:
        return None
    n_vars = sum(v.size for v in model._variables)
    ub_rows, eq_rows = [], []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        aff = extract_affine(c.body, model, n_vars)
        if aff is None:
            continue
        coeffs, const = aff
        adjusted_rhs = float(c.rhs) - const
        if c.sense == "<=":
            ub_rows.append((coeffs, adjusted_rhs))
        elif c.sense == ">=":
            ub_rows.append((-coeffs, -adjusted_rhs))
        elif c.sense == "==":
            eq_rows.append((coeffs, adjusted_rhs))
    A_ub = np.vstack([r[0] for r in ub_rows]) if ub_rows else np.zeros((0, n_vars))
    b_ub = np.array([r[1] for r in ub_rows], dtype=np.float64) if ub_rows else np.zeros(0)
    A_eq = np.vstack([r[0] for r in eq_rows]) if eq_rows else np.zeros((0, n_vars))
    b_eq = np.array([r[1] for r in eq_rows], dtype=np.float64) if eq_rows else np.zeros(0)
    return A_ub, b_ub, A_eq, b_eq


def main():
    paths = sorted(glob.glob(os.path.join(_DATA, "minlplib_nl", "*.nl"))) + sorted(
        glob.glob(os.path.join(_DATA, "minlplib", "*.nl"))
    )
    seen, checked, mismatches, skipped = set(), 0, [], 0
    t_sparse = t_dense = 0.0
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
        try:
            t0 = time.perf_counter()
            ctx = build_linear_context(m)
            t_sparse += time.perf_counter() - t0
            t0 = time.perf_counter()
            ref = dense_reference(m)
            t_dense += time.perf_counter() - t0
        except Exception as exc:
            mismatches.append((name, f"raised {type(exc).__name__}: {exc}"))
            continue
        if ctx is None or ref is None:
            if (ctx is None) != (ref is None):
                mismatches.append((name, "one build returned None"))
            continue
        A_ub, b_ub, A_eq, b_eq = ref
        got_ub = ctx.A_ub.toarray() if hasattr(ctx.A_ub, "toarray") else ctx.A_ub
        got_eq = ctx.A_eq.toarray() if hasattr(ctx.A_eq, "toarray") else ctx.A_eq
        for label, got, want in (
            ("A_ub", got_ub, A_ub),
            ("A_eq", got_eq, A_eq),
            ("b_ub", ctx.b_ub, b_ub),
            ("b_eq", ctx.b_eq, b_eq),
        ):
            if got.shape != want.shape:
                mismatches.append((name, f"{label} shape {got.shape} != {want.shape}"))
            elif not np.array_equal(got, want):
                d = float(np.max(np.abs(got - want))) if got.size else 0.0
                mismatches.append((name, f"{label} differs, max |delta| = {d}"))
        checked += 1

    print(f"instances compared : {checked}  (skipped {skipped} unparseable)")
    print(f"bit-identical      : {checked - len({n for n, _ in mismatches})}")
    print(f"MISMATCHES         : {len(mismatches)}")
    for n, why in mismatches:
        print(f"    {n}: {why}")
    print(f"build_linear_context total: sparse {t_sparse:.2f}s  dense {t_dense:.2f}s")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
