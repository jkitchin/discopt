"""Operator-specific helpers for AMP relaxation builders."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def critical_points_in_interval(start: float, period: float, lb: float, ub: float) -> list[float]:
    tol = 1e-12
    k_min = math.ceil((lb - start - tol) / period)
    k_max = math.floor((ub - start + tol) / period)
    return [start + k * period for k in range(k_min, k_max + 1)]


def tan_range(lb: float, ub: float) -> Optional[tuple[float, float]]:
    """Return tan bounds only when the interval stays on one branch."""
    if not (np.isfinite(lb) and np.isfinite(ub)):
        return None
    margin = 1e-4
    k_min = math.floor((lb - math.pi / 2.0) / math.pi) - 1
    k_max = math.ceil((ub - math.pi / 2.0) / math.pi) + 1
    for k in range(k_min, k_max + 1):
        asymptote = math.pi / 2.0 + k * math.pi
        if lb - margin <= asymptote <= ub + margin:
            return None
    if abs(math.cos(lb)) < 1e-3 or abs(math.cos(ub)) < 1e-3:
        return None
    # The asymptote scan above uses the rounded ``pi/2 + k*pi``, whose error grows
    # with ``|k|``; the sound enclosure decides the branch and rounds outward (#2190).
    from discopt._relax.convexity import interval as iv

    enc = iv.tan(iv.Interval(np.asarray(float(lb)), np.asarray(float(ub))))
    lo, hi = float(enc.lo), float(enc.hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi


def trig_range(func_name: str, lb: float, ub: float) -> Optional[tuple[float, float]]:
    """Sound enclosure of ``sin``/``cos``/``tan`` over ``[lb, ub]`` (``None`` = abstain)."""
    if lb > ub:
        lb, ub = ub, lb
    if func_name == "tan":
        return tan_range(lb, ub)
    if func_name not in ("sin", "cos"):
        return None
    # Sound enclosure (#2190). The inline version this replaced located the
    # extrema with an absolute 1e-12 tolerance on rounded multiples of pi and did
    # not round outward, so it missed the true range by up to ~5e-7 at
    # ``|x| ~ 1e12``.
    from discopt._relax.convexity import interval as iv

    fn = iv.sin if func_name == "sin" else iv.cos
    enc = fn(iv.Interval(np.asarray(float(lb)), np.asarray(float(ub))))
    return float(enc.lo), float(enc.hi)


def trig_value(func_name: str, x: float) -> float:
    if func_name == "sin":
        return float(np.sin(x))
    if func_name == "cos":
        return float(np.cos(x))
    if func_name == "tan":
        return float(np.tan(x))
    raise ValueError(f"Unsupported trigonometric function: {func_name}")


def trig_square_value(func_name: str, x: float) -> float:
    """Evaluate sin(x)^2 or cos(x)^2."""
    value = trig_value(func_name, x)
    return float(value * value)


def trig_square_grad(func_name: str, x: float) -> float:
    """Evaluate the first derivative of sin(x)^2 or cos(x)^2."""
    if func_name == "sin":
        return float(np.sin(2.0 * x))
    if func_name == "cos":
        return float(-np.sin(2.0 * x))
    raise ValueError(f"Unsupported trig-square function: {func_name}")


def trig_square_second_derivative(func_name: str, x: float) -> float:
    """Evaluate the second derivative of sin(x)^2 or cos(x)^2."""
    if func_name == "sin":
        return float(2.0 * np.cos(2.0 * x))
    if func_name == "cos":
        return float(-2.0 * np.cos(2.0 * x))
    raise ValueError(f"Unsupported trig-square function: {func_name}")


def trig_square_range(func_name: str, lb: float, ub: float) -> Optional[tuple[float, float]]:
    """Compute exact continuous bounds for sin(x)^2 or cos(x)^2."""
    if lb > ub:
        lb, ub = ub, lb
    if func_name not in {"sin", "cos"} or not (np.isfinite(lb) and np.isfinite(ub)):
        return None
    if ub - lb >= math.pi:
        return 0.0, 1.0

    points = [lb, ub]
    points.extend(critical_points_in_interval(0.0, math.pi / 2.0, lb, ub))
    vals = [trig_square_value(func_name, p) for p in points if lb - 1e-12 <= p <= ub + 1e-12]
    return min(vals), max(vals)


def trig_square_curvature(func_name: str, lb: float, ub: float) -> Optional[str]:
    """Return certified curvature for sin(x)^2/cos(x)^2, or None if mixed."""
    if func_name not in {"sin", "cos"} or not (np.isfinite(lb) and np.isfinite(ub)):
        return None
    if ub <= lb + 1e-12:
        return None

    tol = 1e-12
    for point in critical_points_in_interval(math.pi / 4.0, math.pi / 2.0, lb, ub):
        if lb + tol < point < ub - tol:
            return None

    midpoint = 0.5 * (lb + ub)
    second = trig_square_second_derivative(func_name, midpoint)
    if second >= -tol:
        return "convex"
    if second <= tol:
        return "concave"
    return None
