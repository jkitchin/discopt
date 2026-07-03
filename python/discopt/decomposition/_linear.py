"""Shared linear-model extraction for the decomposition solvers.

Both Benders and Lagrangian relaxation operate on the model's linear algebra:
a sparse constraint matrix and a linear (minimization) objective. Each row is
tagged with the index of the originating ``model._constraints`` entry, so
Lagrangian relaxation can pick out the rows belonging to a coupling constraint.

The native representation keeps **one row per constraint** with its original
sense (``"<="`` / ``">="`` / ``"=="``); an equality is *not* split into two rows
(so Lagrangian relaxation dualizes it with a single **free** multiplier instead
of two nonnegative ones — half the dual dimension). Consumers that need the
``<=``-canonical form (the Benders recourse LP) call :meth:`LinearModel.rows_leq`,
which expands equalities and flips ``>=`` rows. The matrix is stored sparse
(``scipy.sparse``) so a large model does not materialize an ``O(m·n)`` dense
array; per-block / per-master densification happens on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import scipy.sparse as _sp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    _sp = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

from discopt.modeling.core import Constraint, Model, ObjectiveSense


@dataclass
class LinearModel:
    """Linear model, internally minimizing.

    ``A`` is the native ``(m, n)`` constraint matrix (one row per constraint,
    original sense preserved), a ``scipy.sparse.csr_matrix`` when SciPy is
    available and a dense ``np.ndarray`` otherwise. ``b`` and ``sense`` are the
    per-row right-hand side and sense; ``row_source`` maps each row back to a
    ``model._constraints`` index.
    """

    n: int
    A: Any  # csr_matrix (m, n) or dense ndarray fallback
    b: np.ndarray  # (m,) right-hand side: A x <sense> b
    sense: list[str]  # per row: "<=", ">=", "=="
    row_source: list[int]  # index into model._constraints
    c: np.ndarray  # objective coefficients (minimization), shape (n,)
    c_offset: float
    minimize: bool  # False -> objective was maximize (c already negated)

    @property
    def m(self) -> int:
        return len(self.sense)

    def dense(self) -> np.ndarray:
        """Full native matrix as a dense ``(m, n)`` array."""
        A = self.A
        return A.toarray() if hasattr(A, "toarray") else np.asarray(A, dtype=np.float64)

    def submatrix(self, rows, cols) -> np.ndarray:
        """Dense ``A[rows][:, cols]`` for a block/master partition.

        Densifies only the requested sub-block, never the whole matrix.
        """
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        A = self.A
        if hasattr(A, "tocsr"):
            if rows.size == 0 or cols.size == 0:
                return np.zeros((rows.size, cols.size), dtype=np.float64)
            return A[rows][:, cols].toarray()
        A = np.asarray(A, dtype=np.float64)
        if rows.size == 0 or cols.size == 0:
            return np.zeros((rows.size, cols.size), dtype=np.float64)
        return A[np.ix_(rows, cols)]

    def rows_leq(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """``<=``-canonical dense view ``(A_leq, b_leq, source_leq)``.

        Equalities expand to two rows (``a·x <= b`` and ``-a·x <= -b``) and
        ``>=`` rows flip sign, reproducing the classic Benders row set. The
        returned source list shares an equality's index across its two rows.
        """
        dense = self.dense()
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        src: list[int] = []
        for i, sense in enumerate(self.sense):
            row = dense[i]
            bi = float(self.b[i])
            s = self.row_source[i]
            if sense == "<=":
                rows.append(row)
                rhs.append(bi)
                src.append(s)
            elif sense == ">=":
                rows.append(-row)
                rhs.append(-bi)
                src.append(s)
            else:  # "=="
                rows.append(row)
                rhs.append(bi)
                src.append(s)
                rows.append(-row)
                rhs.append(-bi)
                src.append(s)
        A_leq = np.array(rows) if rows else np.zeros((0, self.n))
        b_leq = np.array(rhs) if rhs else np.zeros(0)
        return A_leq, b_leq, src


def extract_linear(model: Model) -> LinearModel:
    """Extract the native linear rows and a linear objective.

    Raises ``NotImplementedError`` for nonlinear constraints/objectives or
    non-algebraic constraints (SOS, indicator, ...).
    """
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear

    n = sum(v.size for v in model._variables)

    rows: list[np.ndarray] = []
    b_vals: list[float] = []
    senses: list[str] = []
    row_source: list[int] = []

    def _coeffs(expr, what: str):
        # ``_extract_body_coeffs`` raises (e.g. a TypeError on multi-dimensional
        # indexed variables) rather than returning None for some unsupported
        # constructs; normalize every failure to a clean NotImplementedError so
        # the solvers report "unsupported", not a stray internal error.
        try:
            res = _extract_body_coeffs(expr, model, n)
        except Exception as exc:
            raise NotImplementedError(
                f"Decomposition v1 could not extract linear coefficients from {what} "
                "(unsupported construct, e.g. a multi-dimensional indexed variable). "
                "Use a 1-D flattened formulation, or Model.solve()."
            ) from exc
        if res is None:
            raise NotImplementedError(
                f"Decomposition v1 requires a linear {what}; could not extract coefficients."
            )
        return res

    for src, c in enumerate(model._constraints):
        if not isinstance(c, Constraint):
            raise NotImplementedError(
                "Decomposition v1 supports only algebraic linear constraints "
                f"(got {type(c).__name__})."
            )
        if not _is_linear(c.body):
            raise NotImplementedError(
                "Decomposition v1 supports linear constraints only; the model "
                "has a nonlinear constraint."
            )
        vec, off = _coeffs(c.body, "a constraint")
        # Constraint is ``body <sense> 0`` with ``body = vec·x + off``, i.e.
        # ``vec·x <sense> -off``. Keep one row with the original sense.
        rows.append(np.asarray(vec, dtype=np.float64))
        b_vals.append(-float(off))
        senses.append(c.sense)
        row_source.append(src)

    if rows:
        dense = np.array(rows, dtype=np.float64)
        A: object = _sp.csr_matrix(dense) if _HAVE_SCIPY else dense
    else:
        A = _sp.csr_matrix((0, n)) if _HAVE_SCIPY else np.zeros((0, n))
    b = np.array(b_vals, dtype=np.float64) if b_vals else np.zeros(0)

    obj = model._objective
    if obj is None:
        c_vec = np.zeros(n)
        c_off = 0.0
        minimize = True
    else:
        oc = _coeffs(obj.expression, "the objective")
        c_vec = np.asarray(oc[0], dtype=np.float64)
        c_off = float(oc[1])
        minimize = obj.sense == ObjectiveSense.MINIMIZE
    if not minimize:
        c_vec = -c_vec
        c_off = -c_off

    return LinearModel(n, A, b, senses, row_source, c_vec, c_off, minimize)


def relative_gap(ub: float, lb: float) -> float:
    """Relative optimality gap ``(ub - lb) / max(1, |ub|, |lb|)``.

    The ``max(1, ...)`` denominator (rather than ``|ub| + eps``) keeps the gap
    well-scaled when the objective is near zero: a tiny ``|ub|`` would otherwise
    inflate the relative gap and prevent convergence on problems whose optimum is
    close to 0 (T0.5).
    """
    denom = max(1.0, abs(ub), abs(lb))
    return (ub - lb) / denom


def solution_dict(model: Model, x_full: np.ndarray) -> dict[str, np.ndarray]:
    """Split a flat solution vector into a name -> reshaped-array dict."""
    out: dict[str, np.ndarray] = {}
    off = 0
    for v in model._variables:
        vals = x_full[off : off + v.size]
        out[v.name] = np.asarray(vals).reshape(v.shape) if v.shape else vals.reshape(())
        off += v.size
    return out


__all__ = ["LinearModel", "extract_linear", "relative_gap", "solution_dict"]
