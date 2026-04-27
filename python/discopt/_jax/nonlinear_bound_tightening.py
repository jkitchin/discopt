"""Pattern-based nonlinear bound tightening shared across solver frontends.

These rules complement the existing linear FBBT and LP-based OBBT paths.
Each rule must be sound with respect to the current variable box and may only
tighten bounds. The runner clips every rule's output against the current box so
future rules can be added without changing solver-side plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Optional, Sequence

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    VarType,
)

_EFFECTIVE_INF = 1e19


def is_effectively_finite(value: float) -> bool:
    """Return True when a bound is finite in the solver sense."""
    return bool(np.isfinite(value) and abs(float(value)) < _EFFECTIVE_INF)


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
    ) -> tuple[Optional[float], Optional[float]]:
        vals = (numerator / arg_lo, numerator / arg_hi)
        min_val = min(vals)
        max_val = max(vals)
        if rhs >= max_val - 1e-12:
            return None, None
        if rhs < min_val - 1e-12:
            return None, None

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
            vals = (numerator / arg_lo, numerator / arg_hi)
            if rhs < min(vals) - 1e-12:
                _prove_infeasible(
                    self.name,
                    constraint,
                    "reciprocal activity exceeds the upper bound on the sign-stable box",
                )

            arg_lb, arg_ub = self._argument_interval_for_leq(
                numerator,
                rhs,
                arg_lo,
                arg_hi,
            )
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
    MonotoneFunctionBoundsRule(),
    ReciprocalBoundsRule(),
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

    for rule in rules:
        prev_lb = tightened_lb.copy()
        prev_ub = tightened_ub.copy()
        try:
            cand_lb, cand_ub = rule.tighten(model, prev_lb, prev_ub, metadata)
        except NonlinearBoundTighteningInfeasible as exc:
            applied_rules.append(rule.name)
            return (
                prev_lb,
                prev_ub,
                NonlinearBoundTighteningStats(
                    n_tightened=n_tightened,
                    applied_rules=tuple(applied_rules),
                    infeasible=True,
                    infeasibility_reason=str(exc),
                ),
            )

        cand_lb_arr = np.asarray(cand_lb, dtype=np.float64)
        cand_ub_arr = np.asarray(cand_ub, dtype=np.float64)
        empty_indices = np.flatnonzero(cand_lb_arr > cand_ub_arr + 1e-12)
        if empty_indices.size > 0:
            applied_rules.append(rule.name)
            first_idx = int(empty_indices[0])
            return (
                prev_lb,
                prev_ub,
                NonlinearBoundTighteningStats(
                    n_tightened=n_tightened,
                    applied_rules=tuple(applied_rules),
                    infeasible=True,
                    infeasibility_reason=(
                        f"{rule.name} returned an empty interval for flat variable {first_idx}"
                    ),
                ),
            )
        tightened_lb = np.maximum(prev_lb, cand_lb_arr)
        tightened_ub = np.minimum(prev_ub, cand_ub_arr)

        n_changed = int(
            np.count_nonzero(np.abs(tightened_lb - prev_lb) > 1e-12)
            + np.count_nonzero(np.abs(tightened_ub - prev_ub) > 1e-12)
        )
        if n_changed > 0:
            applied_rules.append(rule.name)
            n_tightened += n_changed

    return (
        tightened_lb,
        tightened_ub,
        NonlinearBoundTighteningStats(
            n_tightened=n_tightened,
            applied_rules=tuple(applied_rules),
        ),
    )
