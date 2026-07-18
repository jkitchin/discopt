"""Symbolic differentiation of the rational expression DAG (Tier-2 support).

Tier-2 (convex/KKT) certificates need gradients (KKT stationarity) and Hessians
(convexity) of the objective and constraint bodies. For the *smooth rational*
fragment of the encoding (``const``/``var``/``add``/``sub``/``mul``/``div``/
``pow`` with an integer exponent) the derivative is again a rational-DAG
expression, so it can be evaluated at the incumbent in exact arithmetic by the
same evaluator the feasibility checker uses.

Non-smooth or non-rational nodes (``abs``, transcendental ``fn``) have no
exact-rational derivative here; :func:`differentiate` raises :class:`NotSmooth`
so the checker refuses rather than guesses (Phase 1 handles them over Mathlib
reals).

Nodes are the same tagged dicts the emitter produces (see
``docs/dev/lean-certificate-plan.md`` §expression). Small algebraic
simplifications (drop ``+0``, ``*1``, ``*0``) keep the second-derivative DAGs
from blowing up.
"""

from __future__ import annotations

from typing import Any


class NotSmooth(Exception):
    """Raised when an expression has no exact-rational derivative (abs / fn)."""


def _const(n: int, d: int = 1) -> dict:
    return {"k": "const", "v": [n, d]}


_ZERO = _const(0)
_ONE = _const(1)


def _is_zero(e: dict) -> bool:
    return e.get("k") == "const" and e["v"][0] == 0


def _is_one(e: dict) -> bool:
    return e.get("k") == "const" and e["v"][0] == e["v"][1]


def _add(a: dict, b: dict) -> dict:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return {"k": "add", "l": a, "r": b}


def _sub(a: dict, b: dict) -> dict:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return {"k": "neg", "x": b}
    return {"k": "sub", "l": a, "r": b}


def _mul(a: dict, b: dict) -> dict:
    if _is_zero(a) or _is_zero(b):
        return _ZERO
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    return {"k": "mul", "l": a, "r": b}


def _neg(a: dict) -> dict:
    if _is_zero(a):
        return _ZERO
    return {"k": "neg", "x": a}


def differentiate(node: Any, var: int) -> dict:
    """d(node)/d(x_var) as a rational-DAG expression. Raises :class:`NotSmooth`."""
    k = node["k"]
    if k == "const":
        return _ZERO
    if k == "var":
        return _ONE if node["i"] == var else _ZERO
    if k == "neg":
        return _neg(differentiate(node["x"], var))
    if k == "add":
        return _add(differentiate(node["l"], var), differentiate(node["r"], var))
    if k == "sub":
        return _sub(differentiate(node["l"], var), differentiate(node["r"], var))
    if k == "mul":
        # (l*r)' = l'*r + l*r'
        left, right = node["l"], node["r"]
        return _add(_mul(differentiate(left, var), right), _mul(left, differentiate(right, var)))
    if k == "div":
        # (l/r)' = (l'*r - l*r') / r^2
        left, right = node["l"], node["r"]
        num = _sub(_mul(differentiate(left, var), right), _mul(left, differentiate(right, var)))
        den = {"k": "pow", "l": right, "r": _const(2)}
        return {"k": "div", "l": num, "r": den} if not _is_zero(num) else _ZERO
    if k == "pow":
        # Only a constant integer exponent is smooth-rational here.
        exp = node["r"]
        if exp.get("k") != "const" or exp["v"][1] != 1:
            raise NotSmooth("pow with non-constant / non-integer exponent")
        n = exp["v"][0]
        base = node["l"]
        db = differentiate(base, var)
        if _is_zero(db) or n == 0:
            return _ZERO
        # (base^n)' = n * base^(n-1) * base'
        power = _ONE if n == 1 else {"k": "pow", "l": base, "r": _const(n - 1)}
        return _mul(_mul(_const(n), power), db)
    if k == "abs":
        raise NotSmooth("abs is not differentiable in exact rationals")
    if k == "fn":
        raise NotSmooth(
            f"transcendental function {node.get('name')!r} has no exact-rational derivative"
        )
    raise NotSmooth(f"unknown expression node {k!r}")


def has_variable(node: Any) -> bool:
    """True if the expression references any ``var`` node (i.e. is non-constant)."""
    k = node["k"]
    if k == "var":
        return True
    if k == "const":
        return False
    if k in ("neg", "abs"):
        return has_variable(node["x"])
    if k in ("add", "sub", "mul", "div", "pow"):
        return has_variable(node["l"]) or has_variable(node["r"])
    if k == "fn":
        return any(has_variable(a) for a in node["args"])
    return True  # unknown -> conservatively "depends on x"
