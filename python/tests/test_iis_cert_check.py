"""The IIS engine's certificate check, vectorised (_check_certificates).

It must accept exactly the points the per-key check it replaced accepted
(reproduced below as the reference). The old check rebuilt the final set for
every certificate: O(|IIS|) Python work per point, which on dijle's
6,282-element anytime set ran past the deadline.
"""

import random

import numpy as np
import scipy.sparse as sp
from discopt._iis_milp import _INF, _check_certificates


class _Eng:
    def __init__(self, A, row_lo, row_hi, col_lb, col_ub, row_groups):
        self.A, self.row_lo, self.row_hi = A, row_lo, row_hi
        self.col_lb, self.col_ub, self.row_groups = col_lb, col_ub, row_groups


# ---- reference: the per-key implementation this replaced, verbatim ----
def _violation(x, eng, rows=None, cols_lb=None, cols_ub=None, cols_int=None):
    """Max violation of the given rows / bounds / integralities at ``x``."""
    v = 0.0
    if rows is not None and len(rows):
        act = eng.A[rows] @ x
        lo, hi = eng.row_lo[rows], eng.row_hi[rows]
        v = max(v, float(np.max(np.where(lo > -_INF, lo - act, 0.0), initial=0.0)))
        v = max(v, float(np.max(np.where(hi < _INF, act - hi, 0.0), initial=0.0)))
    if cols_lb is not None and len(cols_lb):
        v = max(v, float(np.max(eng.col_lb[cols_lb] - x[cols_lb], initial=0.0)))
    if cols_ub is not None and len(cols_ub):
        v = max(v, float(np.max(x[cols_ub] - eng.col_ub[cols_ub], initial=0.0)))
    if cols_int is not None and len(cols_int):
        xi = x[cols_int]
        v = max(v, float(np.max(np.abs(xi - np.round(xi)), initial=0.0)))
    return v


def _check_certificate(x, key, eng, cons, bounds, ints, tol) -> bool:
    """``x`` satisfies the final subsystem minus ``key`` and violates ``key``."""
    kind = key[0]
    cons_o = [c for c in cons if not (kind == "constraint" and c == key[1])]
    ints_o = [j for j in ints if not (kind == "integrality" and j == key[1])]
    held = set(ints_o)
    # Bounds of a column held integer apply; for the integrality element itself
    # its own bounds still apply too (relaxing integrality keeps the box).
    lb_cols = sorted(held | {j for j, s in bounds if s == "lower"})
    ub_cols = sorted(held | {j for j, s in bounds if s == "upper"})
    if kind == "integrality":
        lb_cols = sorted(set(lb_cols) | {key[1]})
        ub_cols = sorted(set(ub_cols) | {key[1]})
    if kind == "bound":
        j, s = key[1], key[2]
        if s == "lower":
            lb_cols = [c for c in lb_cols if c != j]
        else:
            ub_cols = [c for c in ub_cols if c != j]
    rows = np.concatenate([eng.row_groups[c] for c in cons_o]) if cons_o else np.zeros(0, int)
    lb_cols = [c for c in lb_cols if eng.col_lb[c] > -_INF]
    ub_cols = [c for c in ub_cols if eng.col_ub[c] < _INF]
    if _violation(x, eng, rows, lb_cols, ub_cols, ints_o) > tol:
        return False
    # ``key`` itself must be violated by more than tol.
    if kind == "constraint":
        e = _violation(x, eng, rows=eng.row_groups[key[1]])
    elif kind == "integrality":
        e = _violation(x, eng, cols_int=[key[1]])
    else:
        j, s = key[1], key[2]
        e = _violation(
            x, eng, cols_lb=[j] if s == "lower" else None, cols_ub=[j] if s == "upper" else None
        )
    return bool(e > tol)


def test_vectorised_check_agrees_with_the_per_key_reference():
    rng = random.Random(11)
    nprng = np.random.default_rng(11)
    checks = accepted = 0
    for _ in range(800):
        n, m = rng.randint(1, 8), rng.randint(1, 8)
        A = sp.csr_matrix(nprng.integers(-2, 3, size=(m, n)).astype(float))
        row_lo = np.where(nprng.random(m) < 0.5, -_INF, nprng.integers(-2, 2, m).astype(float))
        row_hi = np.where(nprng.random(m) < 0.5, _INF, nprng.integers(-1, 3, m).astype(float))
        col_lb = np.where(nprng.random(n) < 0.3, -_INF, nprng.integers(-2, 1, n).astype(float))
        col_ub = np.where(nprng.random(n) < 0.3, _INF, nprng.integers(0, 3, n).astype(float))
        # constraints are groups of rows
        cut = sorted(rng.sample(range(1, m), rng.randint(0, m - 1))) if m > 1 else []
        groups = np.split(np.arange(m), cut)
        row_groups = {c: g for c, g in enumerate(groups)}
        eng = _Eng(A, row_lo, row_hi, col_lb, col_ub, row_groups)
        cons = [c for c in row_groups if rng.random() < 0.7]
        ints = [j for j in range(n) if rng.random() < 0.4]
        bounds = [
            (j, s)
            for j in range(n)
            if j not in ints
            for s, b in (("lower", col_lb[j] > -_INF), ("upper", col_ub[j] < _INF))
            if b and rng.random() < 0.6
        ]
        keys = (
            [("constraint", c) for c in cons]
            + [("integrality", j) for j in ints]
            + [("bound", j, s) for j, s in bounds]
        )
        pending = {}
        for key in keys:
            x = nprng.integers(-3, 4, n).astype(float)
            if rng.random() < 0.5:
                x = x + nprng.choice([0.0, 0.5, 1e-7, 3e-6], n)
            pending[key] = x
        tol = 1e-6
        got = set(_check_certificates(pending, eng, cons, bounds, ints, tol))
        want = {
            k
            for k, x in pending.items()
            if _check_certificate(x, k, eng, sorted(cons), bounds, sorted(ints), tol)
        }
        assert got == want, got ^ want
        checks += len(pending)
        accepted += len(want)
    assert checks > 3000 and accepted > 60
