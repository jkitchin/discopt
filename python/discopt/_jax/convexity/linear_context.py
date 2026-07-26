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
    """Return ``({flat_index: coeff}, const)`` for an affine scalar expression.

    Walks the DAG collecting linear coefficients. Returns ``None``
    when the expression contains any nonlinear operator or shape that
    doesn't reduce to a scalar affine form.

    The walk was always sparse internally; this exposes it, so a caller assembling a
    matrix does not have to round-trip through a dense ``n_vars`` row per constraint
    (issue #875). Indices outside ``[0, n_vars)`` are dropped, exactly as the dense
    form's bounds check did.
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
    return {i: v for i, v in coeffs_dict.items() if 0 <= i < n_vars}, const


def extract_affine(
    expr: Expression, model: Model, n_vars: int
) -> Optional[tuple[np.ndarray, float]]:
    """Dense view of :func:`extract_affine_sparse` — ``(coeffs, const)`` or ``None``.

    Identical coefficients, constant and refusals; only the representation differs.
    Prefer the sparse form in any per-constraint scan that goes on to assemble a
    matrix: see :func:`build_linear_context` for why (issue #875).
    """
    sparse = extract_affine_sparse(expr, model, n_vars)
    if sparse is None:
        return None
    coeffs_dict, const = sparse
    coeffs = np.zeros(n_vars, dtype=np.float64)
    for idx, v in coeffs_dict.items():
        coeffs[idx] = v
    return coeffs, const


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

    ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``lb <= x <= ub``. The
    coefficient matrices may be empty arrays (no linear constraints);
    variable bounds are always present. ``n_vars`` is the flattened
    decision-variable dimension.
    """

    n_vars: int
    lb: np.ndarray
    ub: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
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
        if self.A_ub.size == 0 and self.A_eq.size == 0:
            return lo_box, hi_box

        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_pounce import solve_lp

        # ``bounds`` as (lo, hi) tuples; POUNCE maps ±inf via its own sentinel.
        bounds = [(float(lo), float(hi)) for lo, hi in zip(self.lb, self.ub)]

        A_ub = self.A_ub if self.A_ub.size else None
        b_ub = self.b_ub if self.b_ub.size else None
        A_eq = self.A_eq if self.A_eq.size else None
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

    # Sparse rows, materialised once at the end (#875). The rows are collected as
    # ``({flat_index: coeff}, rhs)`` and scattered into a single pre-allocated
    # ``(m, n)`` array rather than built as ``m`` dense ``n``-vectors and
    # ``np.vstack``-ed.
    #
    # The measurement, because it contradicts the obvious guess. Profiling this
    # function at n_vars=128,000 with 300 rows: ``np.vstack`` was 3.015 s of 3.032 s
    # (99.4%), while all 302 ``np.zeros`` calls together were 0.002 s and the affine
    # walk 0.008 s. The per-row dense allocation is nearly free — ``calloc`` hands
    # back lazily-mapped zero pages — and the cost is ``vstack`` COPYING m x n
    # float64 (307 MB here) and faulting in every one of those pages. Allocating the
    # (m, n) array directly costs 0.0001 s for the same reason.
    #
    # So scattering only the nonzeros makes the assembly O(nnz) instead of O(m * n),
    # and it also keeps peak RSS at the pages actually written rather than the whole
    # matrix. Same output, exactly: entries absent from a row's dict are 0.0 in both
    # forms. On watercontamination0202 this function was 15.8 s over 2 calls, the
    # second-largest phase left in the #875 root setup.
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
            ub_rows.append(({i: -v for i, v in coeffs.items()}, -adjusted_rhs))
        elif c.sense == "==":
            eq_rows.append((coeffs, adjusted_rhs))

    def _materialise(rows: list[tuple[dict[int, float], float]]) -> tuple[np.ndarray, np.ndarray]:
        A = np.zeros((len(rows), n_vars), dtype=np.float64)
        for r, (coeffs, _rhs) in enumerate(rows):
            for i, v in coeffs.items():
                A[r, i] = v
        b = np.array([rhs for _coeffs, rhs in rows], dtype=np.float64)
        return A, b

    A_ub, b_ub = _materialise(ub_rows)
    A_eq, b_eq = _materialise(eq_rows)

    return LinearContext(n_vars=n_vars, lb=lb, ub=ub, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)


__all__ = [
    "LinearContext",
    "build_linear_context",
    "extract_affine",
    "extract_affine_sparse",
]
