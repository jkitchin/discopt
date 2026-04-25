"""Pattern-based nonlinear bound tightening shared across solver frontends.

These rules complement the existing linear FBBT and LP-based OBBT paths.
Each rule must be sound with respect to the current variable box and may only
tighten bounds. The runner clips every rule's output against the current box so
future rules can be added without changing solver-side plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    VarType,
)

_EFFECTIVE_INF = 1e19


def is_effectively_finite(value: float) -> bool:
    """Return True when a bound is finite in the solver sense."""
    return np.isfinite(value) and abs(float(value)) < _EFFECTIVE_INF


@dataclass(frozen=True)
class FlatVariableMetadata:
    """Flat indexing metadata shared by nonlinear tightening rules."""

    base_offsets: dict[int, int]
    flat_var_types: tuple[VarType, ...]

    def scalar_flat_index(self, expr) -> Optional[int]:
        """Return the flat scalar index for a scalar variable expression."""
        if isinstance(expr, Variable):
            if expr.size != 1:
                return None
            return self.base_offsets[id(expr)]

        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            base = expr.base
            base_offset = self.base_offsets[id(base)]
            idx = expr.index
            if base.shape == ():
                flat_idx = 0
            else:
                if not isinstance(idx, tuple):
                    idx = (idx,)
                flat_idx = int(np.ravel_multi_index(idx, base.shape))
            return base_offset + flat_idx

        return None


def build_flat_variable_metadata(model: Model) -> FlatVariableMetadata:
    """Build flat-variable indexing metadata for a model."""
    base_offsets: dict[int, int] = {}
    flat_var_types: list[VarType] = []
    offset = 0
    for var in model._variables:
        base_offsets[id(var)] = offset
        flat_var_types.extend([var.var_type] * var.size)
        offset += var.size
    return FlatVariableMetadata(base_offsets=base_offsets, flat_var_types=tuple(flat_var_types))


@dataclass(frozen=True)
class NonlinearBoundTighteningStats:
    """Summary of a nonlinear tightening pass."""

    n_tightened: int
    applied_rules: tuple[str, ...]


class NonlinearBoundTighteningRule:
    """Base class for extensible nonlinear bound tightening rules."""

    name = "unnamed_rule"

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _constant_value(expr) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1:
        return None
    return float(values[0])


def _match_scaled_linear_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float]]:
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_linear_var(expr.operand, -scale, metadata)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_linear_var(expr.right, scale * left_const, metadata)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_linear_var(expr.left, scale * right_const, metadata)

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is None:
        return None
    return flat_idx, scale


def _flatten_sum(expr, scale: float, out: list[tuple[float, object]]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_sum(expr.left, scale, out)
        _flatten_sum(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_sum(expr.left, scale, out)
        _flatten_sum(expr.right, -scale, out)
        return
    out.append((scale, expr))


def _min_univariate_quadratic(a: float, b: float, lb: float, ub: float) -> float:
    """Return min of a*x^2 + b*x over [lb, ub] for a >= 0."""
    if a < -1e-12:
        raise ValueError("nonconvex univariate quadratic is not supported")

    if abs(a) <= 1e-12:
        return min(b * lb, b * ub)

    x_star = -b / (2.0 * a)
    x_eval = min(max(x_star, lb), ub)
    return min(a * lb * lb + b * lb, a * x_eval * x_eval + b * x_eval, a * ub * ub + b * ub)


def _tighten_univariate_quadratic_interval(
    a: float,
    b: float,
    rhs: float,
    lb: float,
    ub: float,
) -> Optional[tuple[float, float]]:
    """Intersect [lb, ub] with the feasible set of a*x^2 + b*x <= rhs for a >= 0."""
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return (lb, ub) if rhs >= -1e-12 else None
        bound = rhs / b
        if b > 0.0:
            return (lb, min(ub, bound))
        return (max(lb, bound), ub)

    discriminant = b * b + 4.0 * a * rhs
    if discriminant < -1e-12:
        return None
    discriminant = max(discriminant, 0.0)
    sqrt_disc = float(np.sqrt(discriminant))
    root_lo = (-b - sqrt_disc) / (2.0 * a)
    root_hi = (-b + sqrt_disc) / (2.0 * a)
    return (max(lb, root_lo), min(ub, root_hi))


class SumOfSquaresUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from constraints like sum(a_i * x_i^2) <= c."""

    name = "sum_of_squares_upper_bound"

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_square(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_square(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_square(expr.left, scale * right_const, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "**":
            exponent = _constant_value(expr.right)
            if exponent is None or abs(exponent - 2.0) > 1e-12:
                return None
            flat_idx = metadata.scalar_flat_index(expr.left)
            if flat_idx is None:
                return None
            return flat_idx, scale

        return None

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in model._constraints:
            if getattr(constraint, "sense", None) != "<=":
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            square_coeffs: dict[int, float] = {}
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_square(term, scale, metadata)
                if match is None:
                    matches_pattern = False
                    break

                flat_idx, coeff = match
                if coeff <= 0.0:
                    matches_pattern = False
                    break
                square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff

            if not matches_pattern or not square_coeffs:
                continue

            rhs = max(0.0, -constant_term)
            for flat_idx, coeff in square_coeffs.items():
                if coeff <= 0.0:
                    continue

                radius = float(np.sqrt(rhs / coeff))
                new_lb = max(float(tightened_lb[flat_idx]), -radius)
                new_ub = min(float(tightened_ub[flat_idx]), radius)

                var_type = metadata.flat_var_types[flat_idx]
                if var_type == VarType.BINARY:
                    new_lb = max(new_lb, 0.0)
                    new_ub = min(new_ub, 1.0)
                elif var_type == VarType.INTEGER:
                    new_lb = float(np.ceil(new_lb - 1e-9))
                    new_ub = float(np.floor(new_ub + 1e-9))

                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub

        return tightened_lb, tightened_ub


class SeparableQuadraticUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from separable convex quadratic constraints like x + y^2 <= c."""

    name = "separable_quadratic_upper_bound"

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_square(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_square(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_square(expr.left, scale * right_const, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "**":
            exponent = _constant_value(expr.right)
            if exponent is None or abs(exponent - 2.0) > 1e-12:
                return None
            flat_idx = metadata.scalar_flat_index(expr.left)
            if flat_idx is None:
                return None
            return flat_idx, scale

        return None

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()

        for constraint in model._constraints:
            if getattr(constraint, "sense", None) != "<=":
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            quad_coeffs: dict[int, float] = {}
            linear_coeffs: dict[int, float] = {}
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                square_match = self._match_scaled_square(term, scale, metadata)
                if square_match is not None:
                    flat_idx, coeff = square_match
                    quad_coeffs[flat_idx] = quad_coeffs.get(flat_idx, 0.0) + coeff
                    continue

                linear_match = _match_scaled_linear_var(term, scale, metadata)
                if linear_match is not None:
                    flat_idx, coeff = linear_match
                    linear_coeffs[flat_idx] = linear_coeffs.get(flat_idx, 0.0) + coeff
                    continue

                matches_pattern = False
                break

            if not matches_pattern or (not quad_coeffs and not linear_coeffs):
                continue

            coeffs = {
                flat_idx: (
                    quad_coeffs.get(flat_idx, 0.0),
                    linear_coeffs.get(flat_idx, 0.0),
                )
                for flat_idx in set(quad_coeffs) | set(linear_coeffs)
            }
            if any(a < -1e-12 for a, _ in coeffs.values()):
                continue

            min_contribs: dict[int, float] = {}
            for flat_idx, (a, b) in coeffs.items():
                min_contribs[flat_idx] = _min_univariate_quadratic(
                    a,
                    b,
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )

            total_min = float(sum(min_contribs.values()))
            for flat_idx, (a, b) in coeffs.items():
                rhs = -constant_term - (total_min - min_contribs[flat_idx])
                if not np.isfinite(rhs):
                    continue

                interval = _tighten_univariate_quadratic_interval(
                    a,
                    b,
                    rhs,
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )
                if interval is None:
                    continue

                new_lb, new_ub = interval
                var_type = metadata.flat_var_types[flat_idx]
                if var_type == VarType.BINARY:
                    new_lb = max(new_lb, 0.0)
                    new_ub = min(new_ub, 1.0)
                elif var_type == VarType.INTEGER:
                    new_lb = float(np.ceil(new_lb - 1e-9))
                    new_ub = float(np.floor(new_ub + 1e-9))

                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub

        return tightened_lb, tightened_ub


DEFAULT_NONLINEAR_BOUND_RULES: tuple[NonlinearBoundTighteningRule, ...] = (
    SumOfSquaresUpperBoundRule(),
    SeparableQuadraticUpperBoundRule(),
)


def tighten_nonlinear_bounds(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    rules: Sequence[NonlinearBoundTighteningRule] = DEFAULT_NONLINEAR_BOUND_RULES,
) -> tuple[np.ndarray, np.ndarray, NonlinearBoundTighteningStats]:
    """Run registered nonlinear tightening rules on a variable box."""
    tightened_lb = np.asarray(flat_lb, dtype=np.float64).copy()
    tightened_ub = np.asarray(flat_ub, dtype=np.float64).copy()
    metadata = build_flat_variable_metadata(model)

    applied_rules: list[str] = []
    n_tightened = 0

    for rule in rules:
        prev_lb = tightened_lb.copy()
        prev_ub = tightened_ub.copy()
        cand_lb, cand_ub = rule.tighten(model, prev_lb, prev_ub, metadata)
        tightened_lb = np.maximum(prev_lb, np.asarray(cand_lb, dtype=np.float64))
        tightened_ub = np.minimum(prev_ub, np.asarray(cand_ub, dtype=np.float64))

        n_changed = int(
            np.count_nonzero(np.abs(tightened_lb - prev_lb) > 1e-12)
            + np.count_nonzero(np.abs(tightened_ub - prev_ub) > 1e-12)
        )
        if n_changed > 0:
            applied_rules.append(rule.name)
            n_tightened += n_changed

    return tightened_lb, tightened_ub, NonlinearBoundTighteningStats(
        n_tightened=n_tightened,
        applied_rules=tuple(applied_rules),
    )
