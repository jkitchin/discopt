"""Linear-constraint context for constraint-aware sign reasoning.

The syntactic SUSPECT-style walker in :mod:`rules` determines the sign
of each subexpression from its algebraic structure and the declared
variable bounds alone. For atoms with restricted domains (``log``,
``sqrt``, ``1/x``, fractional ``x**p``), a sign of the argument that
is merely ``NONNEG`` or ``UNKNOWN`` is not enough to apply the DCP
concavity / convexity rule. But the argument's sign often *is*
provable once linear inequalities and equalities of the model are
taken into account — for example ``log(1 + x1 - x2)`` with the
constraint ``x2 <= x1`` implies the argument is ``>= 1 > 0``.

This module provides a ``LinearContext`` that holds the model's
linear relaxation (variable bounds + linear inequality and equality
constraints) and can answer range queries on affine expressions via
two pure-Rust POUNCE LP solves. The range is a sound enclosure over the
intersection of the box with the linear relaxation, so the resulting
sign label is mathematically valid as a premise of a DCP rule.

Range enclosures for affine expressions are exact; for nonlinear
arguments we fall back to the declared variable box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# Relative margin widening the POUNCE-IPM affine-range enclosure so it stays a
# sound outer bound despite the interior-point optimum's small tolerance (#356).
_LP_ENCLOSURE_MARGIN = 1e-7

# Per-solve wall-clock cap for the POUNCE affine-range LP. POUNCE's interior-point
# method is pathologically slow on the ill-conditioned LPs some convexity probes
# produce (hda's signomial-equilibrium rows), and convexity analysis issues many
# such probes, so an unbounded solve can hang the whole (CI-visible) analysis.
# Bounding it is sound: a non-optimal result falls back to the box-only enclosure
# below, which is a valid (looser) outer bound — convexity then abstains rather
# than mis-proves, exactly the conservative direction. Well-conditioned probes
# solve far inside this budget (~milliseconds), so proven-convex detection is
# unaffected; only the pathological ill-conditioned probes hit the cap and abstain.
_LP_POUNCE_TIME_LIMIT = 0.5

# Largest linear system for which the POUNCE range LP is attempted at all. Above this
# the box-only enclosure is used instead: sound (a valid, looser outer bound -- see the
# module docstring), and convexity then abstains rather than mis-proving.
#
# Why a size guard is needed at all (#875): the LP is capped per solve, but convexity
# analysis issues MANY range queries, so the cost is (number of probes) x (cap) and the
# cap alone does not bound it. Until the context build was made sparse this was hidden
# -- assembling a dense 106,201 x 106,711 A_eq took 15.7 s and consumed the
# classifier's whole budget before a single range query ran. Making the build 52x
# faster (15.66 s -> 0.30 s) *unlocked* that work and turned a slow classification into
# a hang, which is the failure this guard removes. A speedup that exposes unbounded
# downstream work is not a speedup.
_LP_MAX_ROWS = 20_000


# ──────────────────────────────────────────────────────────────────────
# Affine coefficient extraction
# ──────────────────────────────────────────────────────────────────────


def _compute_var_offset(var: Variable, model: Model) -> int:
    # Delegates to ``Model._flat_var_offset``: a memoized exclusive prefix-sum table,
    # O(1) per lookup, rebuilt only when the (append-only) variable list grows. This
    # re-summed ``model._variables[: var._index]`` from scratch -- O(n_vars) per
    # variable REFERENCE -- which is the quadratic #654 removed on the paths it
    # covered. Pure speedup: ``_variables`` only grows and a Variable's ``_index`` /
    # ``size`` are immutable, so the value is unchanged (#863).
    return model._flat_var_offset(var)


def extract_affine_sparse(
    expr: Expression, model: Model, n_vars: int
) -> Optional[tuple[dict[int, float], float]]:
    """Return ``({col: coeff}, const)`` for an affine scalar expression.

    Walks the DAG collecting linear coefficients. Returns ``None``
    when the expression contains any nonlinear operator or shape that
    doesn't reduce to a scalar affine form.
    """

    def walk(node: Expression, scale: float) -> Optional[tuple[dict[int, float], float]]:
        if isinstance(node, Constant):
            val = np.asarray(node.value)
            if val.ndim == 0:
                return {}, scale * float(val)
            return None

        if isinstance(node, Parameter):
            val = np.asarray(node.value)
            if val.ndim == 0:
                return {}, scale * float(val)
            return None

        if isinstance(node, Variable):
            if node.size != 1:
                return None
            return {_compute_var_offset(node, model): scale}, 0.0

        if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
            base_off = _compute_var_offset(node.base, model)
            idx = node.index
            if isinstance(idx, (int, np.integer)):
                return {base_off + int(idx): scale}, 0.0
            if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], (int, np.integer)):
                return {base_off + int(idx[0]): scale}, 0.0
            try:
                flat = int(np.ravel_multi_index(idx, node.base.shape))
            except (TypeError, ValueError):
                return None
            return {base_off + flat: scale}, 0.0

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                return walk(node.operand, -scale)
            return None

        if isinstance(node, BinaryOp):
            if node.op in ("+", "-"):
                left = walk(node.left, scale)
                right = walk(node.right, scale if node.op == "+" else -scale)
                if left is None or right is None:
                    return None
                merged = dict(left[0])
                for k, v in right[0].items():
                    merged[k] = merged.get(k, 0.0) + v
                return merged, left[1] + right[1]

            if node.op == "*":
                if _is_scalar_const(node.left):
                    return walk(node.right, scale * _scalar_value(node.left))
                if _is_scalar_const(node.right):
                    return walk(node.left, scale * _scalar_value(node.right))
                return None

            if node.op == "/":
                if _is_scalar_const(node.right):
                    divisor = _scalar_value(node.right)
                    if abs(divisor) <= 1e-30:
                        return None
                    return walk(node.left, scale / divisor)
                return None

            return None

        if isinstance(node, SumExpression):
            return walk(node.operand, scale)

        if isinstance(node, SumOverExpression):
            acc: dict[int, float] = {}
            total = 0.0
            for t in node.terms:
                part = walk(t, scale)
                if part is None:
                    return None
                for k, v in part[0].items():
                    acc[k] = acc.get(k, 0.0) + v
                total += part[1]
            return acc, total

        return None

    result = walk(expr, 1.0)
    if result is None:
        return None
    coeffs_dict, const = result
    return {k: v for k, v in coeffs_dict.items() if 0 <= k < n_vars}, const


def extract_affine(
    expr: Expression, model: Model, n_vars: int
) -> Optional[tuple[np.ndarray, float]]:
    """Dense view of :func:`extract_affine_sparse`, kept for callers that want a row.

    Allocates a length-``n_vars`` vector, so it is the wrong entry point for a
    per-constraint scan over a wide model -- that is what cost #875 15.7 s and 23 GB.
    Prefer the sparse core.
    """
    got = extract_affine_sparse(expr, model, n_vars)
    if got is None:
        return None
    coeffs_dict, const = got
    coeffs = np.zeros(n_vars, dtype=np.float64)
    for idx, v in coeffs_dict.items():
        coeffs[idx] = v
    return coeffs, const


def _rows_to_sparse(rows: list[dict[int, float]], n_vars: int) -> sp.csr_matrix:
    """CSR from a list of (col -> coeff) dicts, never materializing a dense row."""
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for r in rows:
        for k in sorted(r):
            v = r[k]
            if v != 0.0:
                indices.append(k)
                data.append(v)
        indptr.append(len(indices))
    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(rows), n_vars),
    )


def _is_scalar_const(expr: Expression) -> bool:
    if isinstance(expr, (Constant, Parameter)):
        val = np.asarray(expr.value)
        return bool(val.ndim == 0)
    return False


def _scalar_value(expr: Expression) -> float:
    return float(np.asarray(expr.value))  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────
# Linear context
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LinearContext:
    """Linear relaxation of a model for affine range queries.

    ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``lb <= x <= ub``. The coefficient
    matrices are scipy CSR and may have zero rows (no linear constraints);
    variable bounds are always present. ``n_vars`` is the flattened
    decision-variable dimension.
    """

    n_vars: int
    lb: np.ndarray
    ub: np.ndarray
    A_ub: "sp.csr_matrix"
    b_ub: np.ndarray
    A_eq: "sp.csr_matrix"
    b_eq: np.ndarray

    def affine_range(self, coeffs: np.ndarray, const: float) -> tuple[float, float]:
        """Sound enclosure of ``coeffs · x + const`` over the relaxation.

        Uses the declared variable bounds as a free box-only enclosure;
        invokes the pure-Rust POUNCE LP only when linear constraints are
        present, since the box-only bound is already optimal otherwise.
        """
        # Box-only enclosure is optimal when there are no linear rows.
        lo_box, hi_box = _box_range(coeffs, self.lb, self.ub)
        lo_box += const
        hi_box += const
        # ``.nnz``, not ``.size``: on a scipy sparse matrix ``.size`` IS nnz, so the
        # two happen to agree here -- but relying on that is the trap recorded in
        # CLAUDE.md (``.size`` means rows x cols on dense, nnz on sparse). Say what is
        # meant: no nonzero rows means the box-only enclosure is already optimal.
        if self.A_ub.nnz == 0 and self.A_eq.nnz == 0:
            return lo_box, hi_box
        # Too big to probe with an LP per query: fall back to the sound box bound.
        if self.A_ub.shape[0] + self.A_eq.shape[0] > _LP_MAX_ROWS:
            return lo_box, hi_box

        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_pounce import solve_lp

        # ``bounds`` as (lo, hi) tuples; POUNCE maps ±inf via its own sentinel.
        bounds = [(float(lo), float(hi)) for lo, hi in zip(self.lb, self.ub)]

        A_ub = self.A_ub if self.A_ub.nnz else None
        b_ub = self.b_ub if self.b_ub.size else None
        A_eq = self.A_eq if self.A_eq.nnz else None
        b_eq = self.b_eq if self.b_eq.size else None

        try:
            lo_res = solve_lp(
                coeffs,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                time_limit=_LP_POUNCE_TIME_LIMIT,
            )
            hi_res = solve_lp(
                -coeffs,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                time_limit=_LP_POUNCE_TIME_LIMIT,
            )
        except (ValueError, RuntimeError, ImportError):
            return lo_box, hi_box

        # POUNCE is an interior-point method, so the reported optimum carries a
        # small tolerance; widen each side by a magnitude-scaled margin so the
        # range stays a *sound* enclosure (lo ≤ true min, hi ≥ true max) rather
        # than risk an over-tight one that would misclassify convexity (#356).
        lo = lo_box
        if lo_res.status == SolveStatus.OPTIMAL and lo_res.objective is not None:
            f = float(lo_res.objective)
            lo = f - _LP_ENCLOSURE_MARGIN * (1.0 + abs(f)) + const
        hi = hi_box
        if hi_res.status == SolveStatus.OPTIMAL and hi_res.objective is not None:
            f = -float(hi_res.objective)
            hi = f + _LP_ENCLOSURE_MARGIN * (1.0 + abs(f)) + const
        # Intersect with the box-only enclosure; LP errors / margins only widen.
        return max(lo, lo_box), min(hi, hi_box)


def _box_range(coeffs: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[float, float]:
    """Box-only enclosure of ``coeffs · x`` without the linear rows."""
    pos = coeffs > 0
    neg = coeffs < 0
    lo = float(np.sum(coeffs[pos] * lb[pos]) + np.sum(coeffs[neg] * ub[neg]))
    hi = float(np.sum(coeffs[pos] * ub[pos]) + np.sum(coeffs[neg] * lb[neg]))
    return lo, hi


def build_linear_context(model: Model) -> Optional[LinearContext]:
    """Assemble a :class:`LinearContext` from a model's linear rows.

    Returns ``None`` when the model has no variables. Nonlinear
    constraints are silently dropped; only constraints that reduce to
    ``coeffs · x + const  sense  0`` contribute rows.
    """
    if not model._variables:
        return None

    n_vars = sum(v.size for v in model._variables)
    lb = np.empty(n_vars, dtype=np.float64)
    ub = np.empty(n_vars, dtype=np.float64)
    offset = 0
    for v in model._variables:
        vlb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        vub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        lb[offset : offset + v.size] = vlb
        ub[offset : offset + v.size] = vub
        offset += v.size

    ub_rows: list[tuple[dict[int, float], float]] = []
    eq_rows: list[tuple[dict[int, float], float]] = []

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        aff = extract_affine_sparse(c.body, model, n_vars)
        if aff is None:
            continue
        coeffs, const = aff
        # body sense rhs  →  (coeffs · x + const)  sense  rhs
        adjusted_rhs = float(c.rhs) - const
        if c.sense == "<=":
            ub_rows.append((coeffs, adjusted_rhs))
        elif c.sense == ">=":
            ub_rows.append(({k: -v for k, v in coeffs.items()}, -adjusted_rhs))
        elif c.sense == "==":
            eq_rows.append((coeffs, adjusted_rhs))

    # Assemble SPARSE. These matrices are extremely thin: on
    # ``watercontamination0202`` (106,711 vars / 107,209 rows) the equality block is
    # 106,201 x 106,711 holding **205,720 nonzeros** -- 0.0018% dense -- and a dense
    # ``vstack`` of it is 90.7 GB of ``nbytes``, which drove peak RSS from 0.38 GB to
    # 23.1 GB and cost 15.7 s of a 30 s ``time_limit`` (#875). Same defect class that
    # #868 removed from the QP extractors and #878 from ``_any_linear_constraint_form``:
    # one dense length-n row materialized per constraint and then kept.
    #
    # ``_rows_to_sparse`` consumes the (col -> coeff) dicts ``extract_affine_sparse``
    # already builds internally, so nothing is densified at any point.
    A_ub = _rows_to_sparse([r[0] for r in ub_rows], n_vars)
    b_ub = np.array([r[1] for r in ub_rows], dtype=np.float64) if ub_rows else np.zeros(0)
    A_eq = _rows_to_sparse([r[0] for r in eq_rows], n_vars)
    b_eq = np.array([r[1] for r in eq_rows], dtype=np.float64) if eq_rows else np.zeros(0)

    return LinearContext(n_vars=n_vars, lb=lb, ub=ub, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)


__all__ = [
    "LinearContext",
    "build_linear_context",
    "extract_affine",
    "extract_affine_sparse",
]
