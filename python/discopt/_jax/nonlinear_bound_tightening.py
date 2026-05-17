"""Pattern-based nonlinear bound tightening shared across solver frontends.

These rules complement the existing linear FBBT and LP-based OBBT paths.
Each rule must be sound with respect to the current variable box and may only
tighten bounds. The runner clips every rule's output against the current box so
future rules can be added without changing solver-side plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NoReturn, Optional, Sequence, TypeVar

import numpy as np

from discopt._jax._numeric import is_effectively_finite as _is_effectively_finite
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    Model,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

is_effectively_finite = _is_effectively_finite

_ScaledMatch = TypeVar("_ScaledMatch")


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
    infeasible: bool = False
    infeasibility_reason: Optional[str] = None


class NonlinearBoundTighteningInfeasible(ValueError):
    """Raised internally when a sound tightening rule proves infeasibility."""


class _ReciprocalIntervalInfeasible:
    """Sentinel for reciprocal interval propagation that proves infeasibility."""


_RECIPROCAL_INTERVAL_INFEASIBLE = _ReciprocalIntervalInfeasible()


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


class _ScaledExpressionMatcher:
    """Shared scale/negation matcher for rule-specific expression leaves."""

    @staticmethod
    def match(
        expr,
        scale: float,
        leaf_match: Callable[[object, float], Optional[_ScaledMatch]],
    ) -> Optional[_ScaledMatch]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return _ScaledExpressionMatcher.match(expr.operand, -scale, leaf_match)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return _ScaledExpressionMatcher.match(expr.right, scale * left_const, leaf_match)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return _ScaledExpressionMatcher.match(expr.left, scale * right_const, leaf_match)
            return None

        return leaf_match(expr, scale)


def _match_scaled_linear_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float]]:
    def leaf_match(leaf: object, leaf_scale: float) -> Optional[tuple[int, float]]:
        flat_idx = metadata.scalar_flat_index(leaf)
        if flat_idx is None:
            return None
        return flat_idx, leaf_scale

    return _ScaledExpressionMatcher.match(expr, scale, leaf_match)


def _match_scaled_square_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float]]:
    def leaf_match(leaf: object, leaf_scale: float) -> Optional[tuple[int, float]]:
        if isinstance(leaf, BinaryOp) and leaf.op == "**":
            exponent = _constant_value(leaf.right)
            if exponent is None or abs(exponent - 2.0) > 1e-12:
                return None
            flat_idx = metadata.scalar_flat_index(leaf.left)
            if flat_idx is None:
                return None
            return flat_idx, leaf_scale
        return None

    return _ScaledExpressionMatcher.match(expr, scale, leaf_match)


def _merge_affine_matches(
    left: tuple[Optional[int], float, float],
    right: tuple[Optional[int], float, float],
) -> Optional[tuple[Optional[int], float, float]]:
    left_idx, left_coeff, left_offset = left
    right_idx, right_coeff, right_offset = right
    if left_idx is not None and right_idx is not None and left_idx != right_idx:
        return None
    flat_idx = left_idx if left_idx is not None else right_idx
    return flat_idx, left_coeff + right_coeff, left_offset + right_offset


def _match_affine_var(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[Optional[int], float, float]]:
    """Match an affine scalar expression a*x + b in one flat variable."""
    const_val = _constant_value(expr)
    if const_val is not None:
        return None, 0.0, scale * const_val

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is not None:
        return flat_idx, scale, 0.0

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_affine_var(expr.operand, -scale, metadata)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_affine_var(expr.right, scale * left_const, metadata)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_affine_var(expr.left, scale * right_const, metadata)
        return None

    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        left = _match_affine_var(expr.left, scale, metadata)
        right_scale = scale if expr.op == "+" else -scale
        right = _match_affine_var(expr.right, right_scale, metadata)
        if left is None or right is None:
            return None
        return _merge_affine_matches(left, right)

    return None


def _merge_single_variable_affine(
    current: Optional[tuple[Optional[int], float, float]],
    candidate: tuple[Optional[int], float, float],
) -> Optional[tuple[Optional[int], float, float]]:
    if current is None:
        return candidate
    return _merge_affine_matches(current, candidate)


def _apply_integrality(
    lb: float,
    ub: float,
    var_type: VarType,
) -> tuple[float, float]:
    if var_type == VarType.BINARY:
        return max(lb, 0.0), min(ub, 1.0)
    if var_type == VarType.INTEGER:
        return float(np.ceil(lb - 1e-9)), float(np.floor(ub + 1e-9))
    return lb, ub


def _constraint_label(constraint) -> str:
    name = getattr(constraint, "name", None)
    return str(name) if name else repr(constraint)


def _prove_infeasible(rule_name: str, constraint, reason: str) -> NoReturn:
    raise NonlinearBoundTighteningInfeasible(
        f"{rule_name} proved infeasibility for {_constraint_label(constraint)}: {reason}"
    )


def _tighten_affine_argument_interval(
    tightened_lb: np.ndarray,
    tightened_ub: np.ndarray,
    metadata: FlatVariableMetadata,
    flat_idx: int,
    coeff: float,
    offset: float,
    arg_lb: Optional[float] = None,
    arg_ub: Optional[float] = None,
) -> None:
    """Intersect a flat variable box with L <= coeff*x + offset <= U."""
    if abs(coeff) <= 1e-12:
        return

    new_lb = float(tightened_lb[flat_idx])
    new_ub = float(tightened_ub[flat_idx])

    if arg_lb is not None and np.isfinite(arg_lb):
        bound = (float(arg_lb) - offset) / coeff
        if coeff > 0.0:
            new_lb = max(new_lb, bound)
        else:
            new_ub = min(new_ub, bound)

    if arg_ub is not None and np.isfinite(arg_ub):
        bound = (float(arg_ub) - offset) / coeff
        if coeff > 0.0:
            new_ub = min(new_ub, bound)
        else:
            new_lb = max(new_lb, bound)

    new_lb, new_ub = _apply_integrality(
        new_lb,
        new_ub,
        metadata.flat_var_types[flat_idx],
    )
    if new_lb <= new_ub:
        tightened_lb[flat_idx] = new_lb
        tightened_ub[flat_idx] = new_ub
        return

    raise NonlinearBoundTighteningInfeasible(
        f"tightened interval is empty for flat variable {flat_idx}: [{new_lb}, {new_ub}]"
    )


def _linearize_affine_expr(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
    n_vars: int,
) -> Optional[tuple[np.ndarray, float]]:
    const_val = _constant_value(expr)
    if const_val is not None:
        return np.zeros(n_vars, dtype=np.float64), scale * const_val

    flat_idx = metadata.scalar_flat_index(expr)
    if flat_idx is not None:
        coeff = np.zeros(n_vars, dtype=np.float64)
        coeff[flat_idx] = scale
        return coeff, 0.0

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _linearize_affine_expr(expr.operand, -scale, metadata, n_vars)

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            left = _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            right = _linearize_affine_expr(expr.right, scale, metadata, n_vars)
            if left is None or right is None:
                return None
            return left[0] + right[0], left[1] + right[1]
        if expr.op == "-":
            left = _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            right = _linearize_affine_expr(expr.right, -scale, metadata, n_vars)
            if left is None or right is None:
                return None
            return left[0] + right[0], left[1] + right[1]
        if expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return _linearize_affine_expr(expr.right, scale * left_const, metadata, n_vars)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return _linearize_affine_expr(expr.left, scale * right_const, metadata, n_vars)
            return None
        if expr.op == "/":
            right_const = _constant_value(expr.right)
            if right_const is not None and abs(right_const) > 1e-12:
                return _linearize_affine_expr(expr.left, scale / right_const, metadata, n_vars)
            return None
        if expr.op == "**":
            exponent = _constant_value(expr.right)
            if exponent == 1.0:
                return _linearize_affine_expr(expr.left, scale, metadata, n_vars)
            if exponent == 0.0:
                return np.zeros(n_vars, dtype=np.float64), scale
            return None

    if isinstance(expr, SumOverExpression):
        coeff = np.zeros(n_vars, dtype=np.float64)
        const = 0.0
        for term in expr.terms:
            piece = _linearize_affine_expr(term, scale, metadata, n_vars)
            if piece is None:
                return None
            coeff += piece[0]
            const += piece[1]
        return coeff, const

    return None


def _affine_bounds(
    coeff: np.ndarray,
    const: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    lower = float(const)
    upper = float(const)
    for c_i, lb_i, ub_i in zip(coeff, lb, ub):
        c = float(c_i)
        if c >= 0.0:
            lower += c * float(lb_i)
            upper += c * float(ub_i)
        else:
            lower += c * float(ub_i)
            upper += c * float(lb_i)
    return lower, upper


def _tighten_affine_upper_bound(
    tightened_lb: np.ndarray,
    tightened_ub: np.ndarray,
    metadata: FlatVariableMetadata,
    coeff: np.ndarray,
    const: float,
    rhs: float,
) -> None:
    """Intersect a box with ``coeff @ x + const <= rhs`` using interval FBBT."""
    for idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue

        other_min = float(const)
        for j, c_j in enumerate(coeff):
            if j == idx:
                continue
            cj = float(c_j)
            if cj >= 0.0:
                other_min += cj * float(tightened_lb[j])
            else:
                other_min += cj * float(tightened_ub[j])

        bound = (float(rhs) - other_min) / c
        new_lb = float(tightened_lb[idx])
        new_ub = float(tightened_ub[idx])
        if c > 0.0:
            new_ub = min(new_ub, bound)
        else:
            new_lb = max(new_lb, bound)

        new_lb, new_ub = _apply_integrality(new_lb, new_ub, metadata.flat_var_types[idx])
        if new_lb <= new_ub:
            tightened_lb[idx] = new_lb
            tightened_ub[idx] = new_ub
            continue

        raise NonlinearBoundTighteningInfeasible(
            f"tightened interval is empty for flat variable {idx}: [{new_lb}, {new_ub}]"
        )


def _flatten_sum(expr, scale: float, out: list[tuple[float, object]]) -> None:
    if isinstance(expr, SumOverExpression):
        for term in expr.terms:
            _flatten_sum(term, scale, out)
        return

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
        return _match_scaled_square_var(expr, scale, metadata)

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
            if getattr(constraint, "sense", None) not in ("<=", "=="):
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

            rhs = -constant_term
            if rhs < -1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "nonnegative sum of squares has a negative upper bound",
                )
            rhs = max(0.0, rhs)
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
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

        return tightened_lb, tightened_ub


class SqrtSumOfSquaresUpperBoundRule(NonlinearBoundTighteningRule):
    """Tighten bounds from constraints like sqrt(sum(a_i * x_i^2)) <= c."""

    name = "sqrt_sum_of_squares_upper_bound"

    def _match_scaled_square(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[int, float]]:
        return _match_scaled_square_var(expr, scale, metadata)

    def _match_sum_of_squares(
        self,
        expr,
        metadata: FlatVariableMetadata,
    ) -> Optional[dict[int, float]]:
        terms: list[tuple[float, object]] = []
        _flatten_sum(expr, 1.0, terms)

        constant_term = 0.0
        square_coeffs: dict[int, float] = {}
        for scale, term in terms:
            const_val = _constant_value(term)
            if const_val is not None:
                constant_term += scale * const_val
                continue

            match = self._match_scaled_square(term, scale, metadata)
            if match is None:
                return None
            flat_idx, coeff = match
            if coeff <= 0.0:
                return None
            square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff

        if abs(constant_term) > 1e-12 or not square_coeffs:
            return None
        return square_coeffs

    def _match_scaled_sqrt_sum_of_squares(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, dict[int, float]]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_sqrt_sum_of_squares(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_sqrt_sum_of_squares(
                    expr.right, scale * left_const, metadata
                )
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_sqrt_sum_of_squares(
                    expr.left, scale * right_const, metadata
                )
            return None

        if not isinstance(expr, FunctionCall) or expr.func_name != "sqrt" or len(expr.args) != 1:
            return None

        square_coeffs = self._match_sum_of_squares(expr.args[0], metadata)
        if square_coeffs is None:
            return None
        return scale, square_coeffs

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
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            sqrt_match: Optional[tuple[float, dict[int, float]]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_sqrt_sum_of_squares(term, scale, metadata)
                if match is None or sqrt_match is not None:
                    matches_pattern = False
                    break
                sqrt_match = match

            if not matches_pattern or sqrt_match is None:
                continue

            sqrt_coeff, square_coeffs = sqrt_match
            if sqrt_coeff <= 0.0:
                continue

            rhs = -constant_term / sqrt_coeff
            if rhs < -1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "nonnegative sqrt sum of squares has a negative upper bound",
                )
            rhs = max(0.0, rhs)
            squared_rhs = rhs * rhs

            for flat_idx, coeff in square_coeffs.items():
                radius = float(np.sqrt(squared_rhs / coeff))
                new_lb = max(float(tightened_lb[flat_idx]), -radius)
                new_ub = min(float(tightened_ub[flat_idx]), radius)
                new_lb, new_ub = _apply_integrality(
                    new_lb,
                    new_ub,
                    metadata.flat_var_types[flat_idx],
                )
                if new_lb <= new_ub:
                    tightened_lb[flat_idx] = new_lb
                    tightened_ub[flat_idx] = new_ub
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

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
        return _match_scaled_square_var(expr, scale, metadata)

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
            if getattr(constraint, "sense", None) not in ("<=", "=="):
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
            if constant_term + total_min > 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "minimum separable quadratic activity exceeds the upper bound",
                )

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
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

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
                else:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"required interval for flat variable {flat_idx} is empty",
                    )

        return tightened_lb, tightened_ub


def _safe_exp(value: float) -> float:
    if value > 709.0:
        return float("inf")
    if value < -745.0:
        return 0.0
    return float(np.exp(value))


def _monotone_function_value(func_name: str, value: float) -> float:
    if func_name == "exp":
        return _safe_exp(value)
    if func_name == "log":
        if value <= 0.0:
            return -float("inf")
        return float(np.log(value))
    if func_name == "log2":
        if value <= 0.0:
            return -float("inf")
        return float(np.log2(value))
    if func_name == "log10":
        if value <= 0.0:
            return -float("inf")
        return float(np.log10(value))
    if func_name == "log1p":
        if value <= -1.0:
            return -float("inf")
        return float(np.log1p(value))
    if func_name == "sqrt":
        if value < 0.0:
            return float("nan")
        return float(np.sqrt(value))
    raise ValueError(f"Unsupported monotone function: {func_name}")


def _inverse_monotone_upper(func_name: str, rhs: float) -> Optional[float]:
    """Return U such that f(arg) <= rhs implies arg <= U."""
    if func_name == "exp":
        if rhs <= 0.0:
            return None
        return float(np.log(rhs))
    if func_name == "log":
        return _safe_exp(rhs)
    if func_name == "log2":
        return float("inf") if rhs > 1024.0 else float(2.0**rhs)
    if func_name == "log10":
        return float("inf") if rhs > 308.0 else float(10.0**rhs)
    if func_name == "log1p":
        return _safe_exp(rhs) - 1.0
    if func_name == "sqrt":
        if rhs < 0.0:
            return None
        return rhs * rhs
    return None


def _inverse_monotone_lower(func_name: str, rhs: float) -> Optional[float]:
    """Return L such that f(arg) >= rhs implies arg >= L."""
    if func_name == "exp":
        if rhs <= 0.0:
            return None
        return float(np.log(rhs))
    if func_name == "log":
        return _safe_exp(rhs)
    if func_name == "log2":
        return 0.0 if rhs < -1074.0 else float(2.0**rhs)
    if func_name == "log10":
        return 0.0 if rhs < -324.0 else float(10.0**rhs)
    if func_name == "log1p":
        return _safe_exp(rhs) - 1.0
    if func_name == "sqrt":
        if rhs <= 0.0:
            return None
        return rhs * rhs
    return None


_MONOTONE_DOMAINS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "exp": (None, None),
    "log": (0.0, None),
    "log2": (0.0, None),
    "log10": (0.0, None),
    "log1p": (-1.0, None),
    "sqrt": (0.0, None),
}


class MonotoneFunctionEqualityRule(NonlinearBoundTighteningRule):
    """Propagate equalities like y == exp(a*x + b) in both directions."""

    name = "monotone_function_equality"

    def _match_scaled_function(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[str, float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_function(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_function(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_function(expr.left, scale * right_const, metadata)
            return None

        if (
            not isinstance(expr, FunctionCall)
            or expr.func_name not in _MONOTONE_DOMAINS
            or len(expr.args) != 1
        ):
            return None

        affine_match = _match_affine_var(expr.args[0], 1.0, metadata)
        if affine_match is None:
            return None
        flat_idx, arg_coeff, arg_offset = affine_match
        if flat_idx is None or abs(arg_coeff) <= 1e-12:
            return None
        return expr.func_name, scale, flat_idx, arg_coeff, arg_offset

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
            if getattr(constraint, "sense", None) != "==":
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            affine_match: Optional[tuple[Optional[int], float, float]] = None
            function_match: Optional[tuple[str, float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_function(term, scale, metadata)
                if match is not None:
                    if function_match is not None:
                        matches_pattern = False
                        break
                    function_match = match
                    continue

                affine_term = _match_affine_var(term, scale, metadata)
                if affine_term is None:
                    matches_pattern = False
                    break
                affine_match = _merge_single_variable_affine(affine_match, affine_term)
                if affine_match is None:
                    matches_pattern = False
                    break

            if (
                not matches_pattern
                or affine_match is None
                or function_match is None
                or affine_match[0] is None
            ):
                continue

            linear_idx, linear_coeff, linear_offset = affine_match
            assert linear_idx is not None
            if abs(linear_coeff) <= 1e-12:
                continue

            func_name, func_coeff, arg_idx, arg_coeff, arg_offset = function_match
            if abs(func_coeff) <= 1e-12:
                continue

            domain_lb, domain_ub = _MONOTONE_DOMAINS[func_name]
            arg_endpoint_a = arg_coeff * float(tightened_lb[arg_idx]) + arg_offset
            arg_endpoint_b = arg_coeff * float(tightened_ub[arg_idx]) + arg_offset
            current_arg_lb = min(arg_endpoint_a, arg_endpoint_b)
            current_arg_ub = max(arg_endpoint_a, arg_endpoint_b)
            if domain_lb is not None:
                current_arg_lb = max(current_arg_lb, domain_lb)
            if domain_ub is not None:
                current_arg_ub = min(current_arg_ub, domain_ub)
            if current_arg_lb > current_arg_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name} argument domain is empty on the current box",
                )

            func_min = _monotone_function_value(func_name, current_arg_lb)
            func_max = _monotone_function_value(func_name, current_arg_ub)

            linear_target_values = (
                -constant_term - func_coeff * func_min,
                -constant_term - func_coeff * func_max,
            )
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                linear_idx,
                linear_coeff,
                linear_offset,
                arg_lb=min(linear_target_values),
                arg_ub=max(linear_target_values),
            )

            linear_endpoint_a = linear_coeff * float(tightened_lb[linear_idx]) + linear_offset
            linear_endpoint_b = linear_coeff * float(tightened_ub[linear_idx]) + linear_offset
            linear_expr_lb = min(linear_endpoint_a, linear_endpoint_b)
            linear_expr_ub = max(linear_endpoint_a, linear_endpoint_b)
            required_values = (
                (-constant_term - linear_expr_lb) / func_coeff,
                (-constant_term - linear_expr_ub) / func_coeff,
            )
            required_func_lb = min(required_values)
            required_func_ub = max(required_values)

            feasible_func_lb = max(required_func_lb, func_min)
            feasible_func_ub = min(required_func_ub, func_max)
            if feasible_func_lb > feasible_func_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name}(argument) range cannot satisfy linked equality",
                )

            arg_lb = domain_lb
            arg_ub = domain_ub
            if np.isfinite(feasible_func_lb):
                lower = _inverse_monotone_lower(func_name, feasible_func_lb)
                if lower is not None:
                    arg_lb = lower if arg_lb is None else max(arg_lb, lower)
            if np.isfinite(feasible_func_ub):
                upper = _inverse_monotone_upper(func_name, feasible_func_ub)
                if upper is None:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be <= {feasible_func_ub}",
                    )
                arg_ub = upper if arg_ub is None else min(arg_ub, upper)

            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                arg_idx,
                arg_coeff,
                arg_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


class MonotoneFunctionBoundsRule(NonlinearBoundTighteningRule):
    """Tighten affine arguments of monotone unary function constraints."""

    name = "monotone_function_bounds"

    def _match_scaled_function(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[str, float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_function(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_function(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_function(expr.left, scale * right_const, metadata)
            return None

        if (
            not isinstance(expr, FunctionCall)
            or expr.func_name not in _MONOTONE_DOMAINS
            or len(expr.args) != 1
        ):
            return None

        affine_match = _match_affine_var(expr.args[0], 1.0, metadata)
        if affine_match is None:
            return None
        flat_idx, arg_coeff, arg_offset = affine_match
        if flat_idx is None or abs(arg_coeff) <= 1e-12:
            return None
        return expr.func_name, scale, flat_idx, arg_coeff, arg_offset

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
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            function_match: Optional[tuple[str, float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_function(term, scale, metadata)
                if match is None or function_match is not None:
                    matches_pattern = False
                    break
                function_match = match

            if not matches_pattern or function_match is None:
                continue

            func_name, func_coeff, flat_idx, arg_coeff, arg_offset = function_match
            if abs(func_coeff) <= 1e-12:
                continue

            domain_lb, domain_ub = _MONOTONE_DOMAINS[func_name]
            arg_lb = domain_lb
            arg_ub = domain_ub
            arg_endpoint_a = arg_coeff * float(tightened_lb[flat_idx]) + arg_offset
            arg_endpoint_b = arg_coeff * float(tightened_ub[flat_idx]) + arg_offset
            current_arg_lb = min(arg_endpoint_a, arg_endpoint_b)
            current_arg_ub = max(arg_endpoint_a, arg_endpoint_b)
            if domain_lb is not None:
                current_arg_lb = max(current_arg_lb, domain_lb)
            if domain_ub is not None:
                current_arg_ub = min(current_arg_ub, domain_ub)
            if current_arg_lb > current_arg_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"{func_name} argument domain is empty on the current box",
                )

            func_min = _monotone_function_value(func_name, current_arg_lb)
            func_max = _monotone_function_value(func_name, current_arg_ub)
            rhs = -constant_term / func_coeff
            if func_coeff > 0.0:
                if rhs < func_min - 1e-12:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be <= {rhs}",
                    )
                upper = _inverse_monotone_upper(func_name, rhs)
                if upper is not None:
                    arg_ub = upper if arg_ub is None else min(arg_ub, upper)
            else:
                if rhs > func_max + 1e-12:
                    _prove_infeasible(
                        self.name,
                        constraint,
                        f"{func_name}(argument) cannot be >= {rhs}",
                    )
                lower = _inverse_monotone_lower(func_name, rhs)
                if lower is not None:
                    arg_lb = lower if arg_lb is None else max(arg_lb, lower)

            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                flat_idx,
                arg_coeff,
                arg_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


class QuadraticEqualityBoundsRule(NonlinearBoundTighteningRule):
    """Propagate equalities like x == y**2 in both directions."""

    name = "quadratic_equality_bounds"

    @staticmethod
    def _square_interval(lb: float, ub: float) -> tuple[float, float]:
        if lb <= 0.0 <= ub:
            return 0.0, max(lb * lb, ub * ub)
        return min(lb * lb, ub * ub), max(lb * lb, ub * ub)

    @staticmethod
    def _intersect_square_preimage(
        current_lb: float,
        current_ub: float,
        square_lb: float,
        square_ub: float,
    ) -> Optional[tuple[float, float]]:
        if square_ub < -1e-12:
            return None
        square_lb = max(0.0, square_lb)
        square_ub = max(0.0, square_ub)

        radius = float(np.sqrt(square_ub))
        new_lb = max(current_lb, -radius)
        new_ub = min(current_ub, radius)

        if square_lb > 1e-12:
            inner = float(np.sqrt(square_lb))
            if new_lb >= 0.0:
                new_lb = max(new_lb, inner)
            elif new_ub <= 0.0:
                new_ub = min(new_ub, -inner)

        if new_lb > new_ub + 1e-12:
            return None
        return new_lb, new_ub

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
            if getattr(constraint, "sense", None) != "==":
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            affine_match: Optional[tuple[Optional[int], float, float]] = None
            square_match: Optional[tuple[int, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = _match_scaled_square_var(term, scale, metadata)
                if match is not None:
                    if square_match is not None:
                        matches_pattern = False
                        break
                    square_match = match
                    continue

                affine_term = _match_affine_var(term, scale, metadata)
                if affine_term is None:
                    matches_pattern = False
                    break
                affine_match = _merge_single_variable_affine(affine_match, affine_term)
                if affine_match is None:
                    matches_pattern = False
                    break

            if (
                not matches_pattern
                or affine_match is None
                or square_match is None
                or affine_match[0] is None
            ):
                continue

            linear_idx, linear_coeff, linear_offset = affine_match
            assert linear_idx is not None
            square_idx, square_coeff = square_match
            if abs(linear_coeff) <= 1e-12 or abs(square_coeff) <= 1e-12:
                continue

            square_min, square_max = self._square_interval(
                float(tightened_lb[square_idx]),
                float(tightened_ub[square_idx]),
            )
            linear_target_values = (
                -constant_term - square_coeff * square_min,
                -constant_term - square_coeff * square_max,
            )
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                linear_idx,
                linear_coeff,
                linear_offset,
                arg_lb=min(linear_target_values),
                arg_ub=max(linear_target_values),
            )

            linear_endpoint_a = linear_coeff * float(tightened_lb[linear_idx]) + linear_offset
            linear_endpoint_b = linear_coeff * float(tightened_ub[linear_idx]) + linear_offset
            linear_expr_lb = min(linear_endpoint_a, linear_endpoint_b)
            linear_expr_ub = max(linear_endpoint_a, linear_endpoint_b)
            required_square_values = (
                (-constant_term - linear_expr_lb) / square_coeff,
                (-constant_term - linear_expr_ub) / square_coeff,
            )
            required_square_lb = min(required_square_values)
            required_square_ub = max(required_square_values)
            feasible_square_lb = max(required_square_lb, square_min, 0.0)
            feasible_square_ub = min(required_square_ub, square_max)
            if feasible_square_lb > feasible_square_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "square term range cannot satisfy linked equality",
                )

            interval = self._intersect_square_preimage(
                float(tightened_lb[square_idx]),
                float(tightened_ub[square_idx]),
                feasible_square_lb,
                feasible_square_ub,
            )
            if interval is None:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {square_idx} is empty",
                )
            new_lb, new_ub = _apply_integrality(
                interval[0],
                interval[1],
                metadata.flat_var_types[square_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[square_idx] = new_lb
                tightened_ub[square_idx] = new_ub
            else:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {square_idx} is empty",
                )

        return tightened_lb, tightened_ub


class SquareDifferenceLowerBoundRule(NonlinearBoundTighteningRule):
    """Tighten the leading square in equalities like ``a*x^2 = b*y^2 + c*z^2``."""

    name = "square_difference_lower_bound"

    def _collect_square_sum(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
        square_coeffs: dict[int, float],
    ) -> Optional[float]:
        const_val = _constant_value(expr)
        if const_val is not None:
            return scale * const_val

        match = _match_scaled_square_var(expr, scale, metadata)
        if match is not None:
            flat_idx, coeff = match
            square_coeffs[flat_idx] = square_coeffs.get(flat_idx, 0.0) + coeff
            return 0.0

        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._collect_square_sum(expr.operand, -scale, metadata, square_coeffs)

        if isinstance(expr, BinaryOp):
            if expr.op == "+":
                left = self._collect_square_sum(expr.left, scale, metadata, square_coeffs)
                right = self._collect_square_sum(expr.right, scale, metadata, square_coeffs)
                if left is None or right is None:
                    return None
                return left + right
            if expr.op == "-":
                left = self._collect_square_sum(expr.left, scale, metadata, square_coeffs)
                right = self._collect_square_sum(expr.right, -scale, metadata, square_coeffs)
                if left is None or right is None:
                    return None
                return left + right
            if expr.op == "*":
                left_const = _constant_value(expr.left)
                if left_const is not None:
                    return self._collect_square_sum(
                        expr.right, scale * left_const, metadata, square_coeffs
                    )
                right_const = _constant_value(expr.right)
                if right_const is not None:
                    return self._collect_square_sum(
                        expr.left, scale * right_const, metadata, square_coeffs
                    )
            if expr.op == "/":
                right_const = _constant_value(expr.right)
                if right_const is not None:
                    return self._collect_square_sum(
                        expr.left, scale / right_const, metadata, square_coeffs
                    )

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
            if getattr(constraint, "sense", None) != "==":
                continue

            square_coeffs: dict[int, float] = {}
            constant = self._collect_square_sum(constraint.body, 1.0, metadata, square_coeffs)
            if constant is None:
                continue
            constant_term = float(constant)

            square_coeffs = {
                idx: coeff for idx, coeff in square_coeffs.items() if abs(coeff) > 1e-12
            }
            negative = [(idx, coeff) for idx, coeff in square_coeffs.items() if coeff < -1e-12]
            positive = [(idx, coeff) for idx, coeff in square_coeffs.items() if coeff > 1e-12]
            if len(negative) != 1 or not positive:
                continue

            target_idx, target_coeff = negative[0]
            target_scale = -target_coeff
            rhs_lb = constant_term
            rhs_ub = constant_term
            for flat_idx, coeff in positive:
                sq_lb, sq_ub = QuadraticEqualityBoundsRule._square_interval(
                    float(tightened_lb[flat_idx]),
                    float(tightened_ub[flat_idx]),
                )
                rhs_lb += coeff * sq_lb
                rhs_ub += coeff * sq_ub

            if rhs_ub < -1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "positive square activity cannot balance the negative square",
                )

            feasible_square_lb = max(0.0, rhs_lb / target_scale)
            feasible_square_ub = max(0.0, rhs_ub / target_scale)
            if feasible_square_lb > feasible_square_ub + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "required square interval is empty",
                )

            interval = QuadraticEqualityBoundsRule._intersect_square_preimage(
                float(tightened_lb[target_idx]),
                float(tightened_ub[target_idx]),
                feasible_square_lb,
                feasible_square_ub,
            )
            if interval is None:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {target_idx} is empty",
                )

            new_lb, new_ub = _apply_integrality(
                interval[0],
                interval[1],
                metadata.flat_var_types[target_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[target_idx] = new_lb
                tightened_ub[target_idx] = new_ub
            else:
                _prove_infeasible(
                    self.name,
                    constraint,
                    f"required interval for flat variable {target_idx} is empty",
                )

        return tightened_lb, tightened_ub


def _match_scaled_constant_division(expr, scale: float) -> Optional[tuple[float, object]]:
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_constant_division(expr.operand, -scale)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_constant_division(expr.right, scale * left_const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_constant_division(expr.left, scale * right_const)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "/":
        return None
    numerator = _constant_value(expr.left)
    if numerator is None or abs(numerator) <= 1e-12:
        return None
    return scale * numerator, expr.right


def _match_scaled_negative_power(
    expr,
    scale: float,
    metadata: FlatVariableMetadata,
) -> Optional[tuple[int, float, float]]:
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_negative_power(expr.operand, -scale, metadata)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_negative_power(expr.right, scale * left_const, metadata)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_negative_power(expr.left, scale * right_const, metadata)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "**":
        return None
    exponent = _constant_value(expr.right)
    if exponent is None or exponent >= 0.0:
        return None
    flat_idx = metadata.scalar_flat_index(expr.left)
    if flat_idx is None:
        return None
    return flat_idx, float(exponent), scale


class PositiveAffineReciprocalBoundsRule(NonlinearBoundTighteningRule):
    """Propagate ``c / positive_affine >= rhs`` into affine upper bounds."""

    name = "positive_affine_reciprocal_bounds"

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        n_vars = len(flat_lb)

        for constraint in model._constraints:
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            reciprocal_match: Optional[tuple[float, object]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = _match_scaled_constant_division(term, scale)
                if match is None or reciprocal_match is not None:
                    matches_pattern = False
                    break
                reciprocal_match = match

            if not matches_pattern or reciprocal_match is None or constant_term <= 0.0:
                continue

            scaled_numerator, denominator = reciprocal_match
            if scaled_numerator >= 0.0:
                continue

            affine = _linearize_affine_expr(denominator, 1.0, metadata, n_vars)
            if affine is None:
                continue
            coeff, const = affine
            arg_lb, arg_ub = _affine_bounds(coeff, const, tightened_lb, tightened_ub)
            if arg_lb <= 0.0:
                continue

            rhs = -scaled_numerator / constant_term
            if not np.isfinite(rhs):
                continue
            if arg_lb > rhs + 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "positive affine denominator exceeds reciprocal upper bound",
                )
            if arg_ub <= rhs + 1e-12:
                continue
            _tighten_affine_upper_bound(tightened_lb, tightened_ub, metadata, coeff, const, rhs)

        return tightened_lb, tightened_ub


class NegativePowerBoundsRule(NonlinearBoundTighteningRule):
    """Infer strict positive lower bounds from ``x**p <= affine`` for ``p < 0``."""

    name = "negative_power_bounds"

    def tighten(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        metadata: FlatVariableMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        tightened_lb = flat_lb.copy()
        tightened_ub = flat_ub.copy()
        n_vars = len(flat_lb)

        for constraint in model._constraints:
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            power_match: Optional[tuple[int, float, float]] = None
            affine_coeff = np.zeros(n_vars, dtype=np.float64)
            affine_const = 0.0
            matches_pattern = True

            for scale, term in terms:
                match = _match_scaled_negative_power(term, scale, metadata)
                if match is not None:
                    if power_match is not None:
                        matches_pattern = False
                        break
                    power_match = match
                    continue

                affine = _linearize_affine_expr(term, scale, metadata, n_vars)
                if affine is None:
                    matches_pattern = False
                    break
                affine_coeff += affine[0]
                affine_const += affine[1]

            if not matches_pattern or power_match is None:
                continue

            base_idx, exponent, power_scale = power_match
            if power_scale <= 0.0:
                continue
            if abs(float(affine_coeff[base_idx])) > 1e-12:
                continue
            if tightened_lb[base_idx] < -1e-12 or tightened_ub[base_idx] <= 0.0:
                continue

            affine_min, _affine_max = _affine_bounds(
                affine_coeff,
                affine_const,
                tightened_lb,
                tightened_ub,
            )
            rhs_ub = -affine_min / power_scale
            if rhs_ub <= 0.0:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "negative power has no positive upper allowance",
                )
            if not np.isfinite(rhs_ub):
                continue

            lower = float(rhs_ub ** (1.0 / exponent))
            if not np.isfinite(lower):
                continue
            new_lb = max(float(tightened_lb[base_idx]), lower)
            new_ub = float(tightened_ub[base_idx])
            new_lb, new_ub = _apply_integrality(
                new_lb,
                new_ub,
                metadata.flat_var_types[base_idx],
            )
            if new_lb <= new_ub:
                tightened_lb[base_idx] = new_lb
                tightened_ub[base_idx] = new_ub
                continue
            _prove_infeasible(
                self.name,
                constraint,
                f"required positive lower bound {new_lb} exceeds upper bound {new_ub}",
            )

        return tightened_lb, tightened_ub


class ReciprocalBoundsRule(NonlinearBoundTighteningRule):
    """Tighten sign-stable affine denominators in simple reciprocal constraints."""

    name = "reciprocal_bounds"

    def _match_scaled_reciprocal(
        self,
        expr,
        scale: float,
        metadata: FlatVariableMetadata,
    ) -> Optional[tuple[float, int, float, float]]:
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            return self._match_scaled_reciprocal(expr.operand, -scale, metadata)

        if isinstance(expr, BinaryOp) and expr.op == "*":
            left_const = _constant_value(expr.left)
            if left_const is not None:
                return self._match_scaled_reciprocal(expr.right, scale * left_const, metadata)
            right_const = _constant_value(expr.right)
            if right_const is not None:
                return self._match_scaled_reciprocal(expr.left, scale * right_const, metadata)
            return None

        if not isinstance(expr, BinaryOp) or expr.op != "/":
            return None

        numerator = _constant_value(expr.left)
        if numerator is None or abs(numerator) <= 1e-12:
            return None

        denominator = _match_affine_var(expr.right, 1.0, metadata)
        if denominator is None:
            return None
        flat_idx, denom_coeff, denom_offset = denominator
        if flat_idx is None or abs(denom_coeff) <= 1e-12:
            return None

        return scale * numerator, flat_idx, denom_coeff, denom_offset

    @staticmethod
    def _argument_interval_for_leq(
        numerator: float,
        rhs: float,
        arg_lo: float,
        arg_hi: float,
    ) -> Optional[tuple[Optional[float], Optional[float]] | _ReciprocalIntervalInfeasible]:
        vals = (numerator / arg_lo, numerator / arg_hi)
        min_val = min(vals)
        max_val = max(vals)
        if rhs >= max_val - 1e-12:
            return None
        if rhs < min_val - 1e-12:
            return _RECIPROCAL_INTERVAL_INFEASIBLE

        threshold = numerator / rhs
        if numerator > 0.0:
            return threshold, None
        return None, threshold

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
            if getattr(constraint, "sense", None) not in ("<=", "=="):
                continue

            terms: list[tuple[float, object]] = []
            _flatten_sum(constraint.body, 1.0, terms)

            constant_term = 0.0
            reciprocal_match: Optional[tuple[float, int, float, float]] = None
            matches_pattern = True

            for scale, term in terms:
                const_val = _constant_value(term)
                if const_val is not None:
                    constant_term += scale * const_val
                    continue

                match = self._match_scaled_reciprocal(term, scale, metadata)
                if match is None or reciprocal_match is not None:
                    matches_pattern = False
                    break
                reciprocal_match = match

            if not matches_pattern or reciprocal_match is None:
                continue

            numerator, flat_idx, denom_coeff, denom_offset = reciprocal_match
            arg_endpoint_a = denom_coeff * float(tightened_lb[flat_idx]) + denom_offset
            arg_endpoint_b = denom_coeff * float(tightened_ub[flat_idx]) + denom_offset
            arg_lo = min(arg_endpoint_a, arg_endpoint_b)
            arg_hi = max(arg_endpoint_a, arg_endpoint_b)
            if arg_lo <= 0.0 <= arg_hi:
                continue

            rhs = -constant_term
            arg_interval = self._argument_interval_for_leq(
                numerator,
                rhs,
                arg_lo,
                arg_hi,
            )
            if isinstance(arg_interval, _ReciprocalIntervalInfeasible):
                _prove_infeasible(
                    self.name,
                    constraint,
                    "reciprocal activity exceeds the upper bound on the sign-stable box",
                )
            elif arg_interval is None:
                continue
            else:
                arg_lb, arg_ub = arg_interval
            _tighten_affine_argument_interval(
                tightened_lb,
                tightened_ub,
                metadata,
                flat_idx,
                denom_coeff,
                denom_offset,
                arg_lb=arg_lb,
                arg_ub=arg_ub,
            )

        return tightened_lb, tightened_ub


DEFAULT_NONLINEAR_BOUND_RULES: tuple[NonlinearBoundTighteningRule, ...] = (
    MonotoneFunctionEqualityRule(),
    QuadraticEqualityBoundsRule(),
    SquareDifferenceLowerBoundRule(),
    MonotoneFunctionBoundsRule(),
    PositiveAffineReciprocalBoundsRule(),
    NegativePowerBoundsRule(),
    ReciprocalBoundsRule(),
    SqrtSumOfSquaresUpperBoundRule(),
    SumOfSquaresUpperBoundRule(),
    SeparableQuadraticUpperBoundRule(),
)


def tighten_nonlinear_bounds(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    rules: Sequence[NonlinearBoundTighteningRule] = DEFAULT_NONLINEAR_BOUND_RULES,
    max_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, NonlinearBoundTighteningStats]:
    """Run registered nonlinear tightening rules on a variable box."""
    tightened_lb = np.asarray(flat_lb, dtype=np.float64).copy()
    tightened_ub = np.asarray(flat_ub, dtype=np.float64).copy()
    initial_lb = tightened_lb.copy()
    initial_ub = tightened_ub.copy()
    metadata = build_flat_variable_metadata(model)

    applied_rules: list[str] = []

    def _mark_rule(rule_name: str) -> None:
        if rule_name not in applied_rules:
            applied_rules.append(rule_name)

    def _count_changed(new: np.ndarray, old: np.ndarray) -> int:
        finite = np.isfinite(new) & np.isfinite(old)
        changed = np.zeros(new.shape, dtype=bool)
        changed[finite] = np.abs(new[finite] - old[finite]) > 1e-12
        changed[~finite] = new[~finite] != old[~finite]
        return int(np.count_nonzero(changed))

    def _count_tightened(lb: np.ndarray, ub: np.ndarray) -> int:
        return int(_count_changed(lb, initial_lb) + _count_changed(ub, initial_ub))

    empty_initial = np.flatnonzero(tightened_lb > tightened_ub + 1e-12)
    if empty_initial.size > 0:
        first_idx = int(empty_initial[0])
        return (
            tightened_lb,
            tightened_ub,
            NonlinearBoundTighteningStats(
                n_tightened=0,
                applied_rules=(),
                infeasible=True,
                infeasibility_reason=f"initial interval is empty for flat variable {first_idx}",
            ),
        )

    for _ in range(max(1, int(max_rounds))):
        round_changed = False
        for rule in rules:
            prev_lb = tightened_lb.copy()
            prev_ub = tightened_ub.copy()
            try:
                cand_lb, cand_ub = rule.tighten(model, prev_lb, prev_ub, metadata)
            except NonlinearBoundTighteningInfeasible as exc:
                _mark_rule(rule.name)
                return (
                    prev_lb,
                    prev_ub,
                    NonlinearBoundTighteningStats(
                        n_tightened=_count_tightened(prev_lb, prev_ub),
                        applied_rules=tuple(applied_rules),
                        infeasible=True,
                        infeasibility_reason=str(exc),
                    ),
                )

            cand_lb_arr = np.asarray(cand_lb, dtype=np.float64)
            cand_ub_arr = np.asarray(cand_ub, dtype=np.float64)
            empty_indices = np.flatnonzero(cand_lb_arr > cand_ub_arr + 1e-12)
            if empty_indices.size > 0:
                _mark_rule(rule.name)
                first_idx = int(empty_indices[0])
                return (
                    prev_lb,
                    prev_ub,
                    NonlinearBoundTighteningStats(
                        n_tightened=_count_tightened(prev_lb, prev_ub),
                        applied_rules=tuple(applied_rules),
                        infeasible=True,
                        infeasibility_reason=(
                            f"{rule.name} returned an empty interval for flat variable {first_idx}"
                        ),
                    ),
                )
            tightened_lb = np.maximum(prev_lb, cand_lb_arr)
            tightened_ub = np.minimum(prev_ub, cand_ub_arr)

            n_changed = _count_changed(tightened_lb, prev_lb) + _count_changed(
                tightened_ub, prev_ub
            )
            if n_changed > 0:
                _mark_rule(rule.name)
                round_changed = True
        if not round_changed:
            break

    return (
        tightened_lb,
        tightened_ub,
        NonlinearBoundTighteningStats(
            n_tightened=_count_tightened(tightened_lb, tightened_ub),
            applied_rules=tuple(applied_rules),
        ),
    )
