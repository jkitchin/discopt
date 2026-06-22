"""Aggregation / complemented mixed-integer rounding (c-MIR) cut separator.

discopt's existing single-row MIR (and GMI) plateau on the McCormick LP of an
integer-product MINLP: they separate a vertex but not the optimal face, so the
dual bound barely moves (see docs/dev/scip-gap-nvs-diagnosis.md). SCIP closes the
same instances with its ``aggregation`` separator — *multi-row complemented MIR*
(Marchand & Wolsey, "Aggregation and mixed integer rounding to solve MIPs",
Oper. Res. 2001). This module implements that procedure:

1. **Aggregation** — combine several ``a·x <= b`` rows with nonnegative weights into
   one valid mixed-integer row (here: each single row, plus dual-weighted and
   pairwise aggregations of the rows binding at the LP optimum).
2. **Bound substitution / complementation** — replace each variable by its distance
   from the *nearer* bound (``x_j - l_j`` or ``u_j - x_j``); the complementation
   choice is the extra degree of freedom plain MIR lacks and is what lets the cut
   reach the optimal face.
3. **Scaling + MIR** — divide by ``delta`` (1 and ``|a_j|`` for integer columns) and
   apply the MIR function, keeping the most violated valid cut.

Every produced inequality is a valid MIR inequality of a nonnegative aggregation of
valid rows, hence valid for the integer-feasible set; a small violation threshold
plus integer snapping keeps it from cutting a feasible point on floating-point
noise. The separator only ever tightens — it cannot change the optimum.
"""

from __future__ import annotations

from math import floor
from typing import Optional

import numpy as np

_SNAP = 1e-9
_F0_LO, _F0_HI = 1e-4, 1 - 1e-4
_VIOL = 1e-5
_MAX_COEF = 1e7


def _snap(v: np.ndarray) -> np.ndarray:
    r = np.round(v)
    return np.where(np.abs(v - r) <= _SNAP, r, v)


def _cmir_row(a, b, xstar, lb, ub, is_int) -> Optional[tuple[np.ndarray, float, float]]:
    """Complemented-MIR on one aggregated row ``a·x <= b``. Returns
    ``(coeffs, rhs, violation)`` for a cut ``coeffs·x <= rhs`` violated at
    ``xstar``, or ``None``. Variables must be bounded (finite lb/ub)."""
    a = _snap(np.asarray(a, dtype=np.float64))
    width = ub - lb
    # complement variables sitting nearer their upper bound -> y_j = u_j - x_j >= 0,
    # else y_j = x_j - l_j >= 0. y* is then small, which strengthens the MIR cut.
    comp = xstar > 0.5 * (lb + ub)
    atil = np.where(comp, -a, a)
    btil = float(b - np.sum(np.where(comp, a * ub, a * lb)))
    ystar = np.where(comp, ub - xstar, xstar - lb)

    deltas = [1.0]
    deltas += sorted(
        {abs(float(atil[j])) for j in range(len(a)) if is_int[j] and abs(atil[j]) > _SNAP}
    )
    best = None
    for d in deltas:
        if d < _SNAP:
            continue
        ar = _snap(atil / d)
        br = btil / d
        f0 = br - floor(br)
        if f0 < _F0_LO or f0 > _F0_HI:
            continue
        gamma = np.empty(len(a), dtype=np.float64)
        for j in range(len(a)):
            if is_int[j]:
                fj = ar[j] - floor(ar[j])
                gamma[j] = floor(ar[j]) + max(0.0, (fj - f0) / (1.0 - f0))
            else:
                gamma[j] = min(ar[j], 0.0) / (1.0 - f0)
        rhs = float(floor(br))
        if np.max(np.abs(gamma)) > _MAX_COEF:
            continue
        viol = float(gamma @ ystar) - rhs
        # also require the cut to be (weakly) violated in y>=0 space: bounded width
        if viol > _VIOL and (best is None or viol > best[2]):
            best = (gamma.copy(), rhs, viol)
    if best is None:
        return None
    gamma, rhs, viol = best
    # un-complement back to x:  gamma·y = sum( comp? gamma_j(u_j-x_j) : gamma_j(x_j-l_j) )
    cx = np.where(comp, -gamma, gamma)
    const = float(np.sum(np.where(comp, gamma * ub, -gamma * lb)))
    crhs = rhs - const
    # margin so floating-point error can't cut a feasible integer point
    margin = 1e-7 * (1.0 + float(np.abs(cx).sum()))
    return cx, crhs + margin, viol


def separate_cmir(
    A, b, xstar, lb, ub, is_int, *, max_cuts: int = 16, duals: Optional[np.ndarray] = None
):
    """Separate complemented-MIR cuts from ``A x <= b`` at ``xstar``.

    Aggregations tried: every single row; a dual-weighted aggregation of the rows
    binding at ``xstar`` (if ``duals`` given); and pairwise sums of binding rows.
    Returns a list of ``(coeffs, rhs)`` cuts ``coeffs·x <= rhs`` (most-violated
    first, de-duplicated), each valid for the integer-feasible set."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    xstar = np.asarray(xstar, dtype=np.float64).ravel()
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    is_int = np.asarray(is_int, dtype=bool).ravel()
    m = A.shape[0]
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        return []

    rows: list[tuple[np.ndarray, float]] = []
    # 1) single rows
    for r in range(m):
        rows.append((A[r], float(b[r])))
    # binding rows at xstar
    resid = b - A @ xstar
    binding = [r for r in range(m) if abs(resid[r]) < 1e-6]
    # 2) dual-weighted aggregation of binding rows
    if duals is not None and len(binding) > 1:
        w = np.asarray(duals, dtype=np.float64).ravel()
        if w.shape[0] == m:
            agg_a = np.zeros(A.shape[1])
            agg_b = 0.0
            for r in binding:
                wr = abs(float(w[r]))
                agg_a += wr * A[r]
                agg_b += wr * b[r]
            rows.append((agg_a, agg_b))
    # 3) pairwise aggregations of binding rows (bounded count)
    for ii in range(min(len(binding), 8)):
        for jj in range(ii + 1, min(len(binding), 8)):
            r1, r2 = binding[ii], binding[jj]
            rows.append((A[r1] + A[r2], float(b[r1] + b[r2])))

    found: list[tuple[float, tuple[float, ...], np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    for a_row, b_row in rows:
        res = _cmir_row(a_row, b_row, xstar, lb, ub, is_int)
        if res is None:
            continue
        cx, crhs, viol = res
        key = tuple(np.round(cx, 5))
        if key in seen:
            continue
        seen.add(key)
        found.append((viol, key, cx, crhs))
    found.sort(key=lambda t: -t[0])
    return [(cx, crhs) for _v, _k, cx, crhs in found[:max_cuts]]
