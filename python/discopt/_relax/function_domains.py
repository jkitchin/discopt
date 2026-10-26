"""Linear rows implied by the domains of a model's functions (#1979).

``log(t)``, ``log2``, ``log10`` and ``sqrt(t)`` are real only for ``t >= 0`` (``log``
for ``t > 0``), ``log1p(t)`` for ``t >= -1``, and ``t ** p`` with a constant
non-integer ``p`` for ``t >= 0``. A model is evaluable -- and a point can be its
optimum -- only where those hold, the same semantics the interval rules already
assume (``interval_pow`` clamps the base; the backward ``log`` rule maps ``(-inf, hi]``
to ``[0, e^hi]``). When the argument ``t`` is AFFINE in the model's variables, the
domain is a linear row ``t >= lo``, and handing it to the LP relaxation and to OBBT
lets them use it: on ``ex8_5_*`` the objective's ``log(x3 - x5)`` gives
``x3 - x5 >= 0``, which with the model's linear rows (``x5 = 0.14998 * (x0 + x1 + x2)``,
``x0 + x1 + x2 = 1``) puts ``x3 >= 0.14998`` -- the bound that moves the objective's
``x4 / x3`` pole off the box. A single-variable argument is a plain bound, which the
FBBT domain pass (``presolve/fbbt.rs`` ``impose_function_domains``) already applies,
so only arguments over two or more variables are returned.

Gated, with the FBBT sign/domain rules, by ``DISCOPT_FBBT_SIGN_DOMAIN`` (default ON
since its bundle graduation; ``=0`` opts out).
"""

from __future__ import annotations

import os

import numpy as np

from discopt.modeling.core import BinaryOp, Constant, Expression, FunctionCall, Model

#: Lower edge of each function's argument domain.
_DOMAIN_LO = {"log": 0.0, "log2": 0.0, "log10": 0.0, "sqrt": 0.0, "log1p": -1.0}


def function_domains_enabled() -> bool:
    """``DISCOPT_FBBT_SIGN_DOMAIN``: read at call time. Default ON (#1982 bundle graduation)."""
    return os.environ.get("DISCOPT_FBBT_SIGN_DOMAIN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _domain_arg(node: Expression):
    """``(argument, lo)`` when ``node`` requires ``argument >= lo``, else ``None``."""
    if isinstance(node, FunctionCall) and node.func_name in _DOMAIN_LO and node.args:
        return node.args[0], _DOMAIN_LO[node.func_name]
    if isinstance(node, BinaryOp) and node.op == "**" and isinstance(node.right, Constant):
        p = np.asarray(node.right.value)
        if p.ndim == 0 and np.isfinite(float(p)) and abs(float(p) - round(float(p))) >= 1e-12:
            return node.left, 0.0
    return None


def affine_domain_rows(model: Model) -> list[tuple[dict[int, float], float, float]]:
    """``[(terms, const, lo)]``, each meaning ``terms . x + const >= lo``.

    One row per distinct affine argument over two or more variables found anywhere
    in the objective or a constraint body. A non-affine argument, or one whose
    coefficients are not finite, is skipped (dropping a row only loosens). Cached on
    the model, keyed by its objective and constraint list (a relaxation is built per
    node).
    """
    key = (id(model._objective), len(model._constraints))
    cached = getattr(model, "_function_domain_rows", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    rows = _affine_domain_rows(model)
    model._function_domain_rows = (key, rows)
    return rows


def _affine_domain_rows(model: Model) -> list[tuple[dict[int, float], float, float]]:
    from discopt._relax.poles import _children
    from discopt._relax.problem_classifier import (
        _extract_linear_coefficients_sparse,
        _NotLinearError,
    )

    n = sum(int(v.size) for v in model._variables)
    roots: list[Expression] = [c.body for c in model._constraints if hasattr(c, "body")]
    if model._objective is not None:
        roots.append(model._objective.expression)
    rows: dict[tuple, tuple[dict[int, float], float, float]] = {}
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        dom = _domain_arg(node)
        if dom is not None:
            arg, lo = dom
            try:
                terms, const = _extract_linear_coefficients_sparse(arg, model, n)
            except (_NotLinearError, ValueError, TypeError):
                terms = None
            if terms is not None:
                terms = {int(j): float(a) for j, a in terms.items() if a != 0.0}
                ok = len(terms) >= 2 and np.isfinite(const)
                if ok and all(np.isfinite(a) for a in terms.values()):
                    key = (tuple(sorted(terms.items())), float(const), float(lo))
                    rows[key] = (terms, float(const), float(lo))
        stack.extend(_children(node))
    return list(rows.values())
