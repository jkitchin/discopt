"""Objective-defining-equality relaxation (the SUSPECT "objective constraint").

Many MINLP models — especially MINLPLib / GAMS instances — are written as

    minimize  z
    s.t.      z = g(x)            (a single equality "defining" the objective)
              other constraints not involving z

where ``z`` is a *free* scalar variable that appears in exactly one
constraint, affinely. When ``g`` is convex this is a convex problem in
disguise, yet the equality ``z = g(x)`` has a *non-convex* feasible set
(an equality is convex only when its body is affine), so a syntactic
convexity check correctly rejects it and the solver falls back to
nonconvex spatial branch-and-bound — with no valid lower bound and an
erratic NLP multistart incumbent (issue: du-opt).

The classical fix (BARON, SCIP, and SUSPECT all do this internally) is to
relax the *defining equality* to the inequality the objective binds
against:

    minimize z   s.t.  z >= g(x)        (for ``min``, ``z`` free below)

This relaxation is **exact at the optimum**: at any optimum ``(z*, x*)``
of the relaxed problem with ``z* > g(x*)`` we could lower ``z*`` to
``g(x*)`` — still feasible (``z >= g`` holds), the other constraints do
not involve ``z``, and the objective strictly improves — contradicting
optimality. Hence ``z* = g(x*)``: the relaxed optimum is feasible for the
original equality and has the same objective value. The argument needs no
assumption on the curvature of ``g`` (it is exact for convex *and*
nonconvex ``g``); convexity only governs whether the *relaxed* constraint
is itself convex and therefore unlocks the convex solve path.

Soundness invariant
-------------------
The transform fires only when the rewrite is provably exact *and* turns
the constraint convex:

1. The objective is exactly a single scalar variable ``z`` (a maximize or
   minimize of ``z``).
2. ``z`` is free in the binding direction (``lb <= -1e15`` for ``min``,
   ``ub >= 1e15`` for ``max``) so the objective can always drive the
   inequality tight.
3. ``z`` appears in exactly one constraint, which is an equality, and in
   that constraint ``z`` appears *affinely* with a constant nonzero
   coefficient (a structural proof, not sampling — see
   :func:`_affine_coeff`). ``z`` appears in no other constraint.
4. The body is genuinely curved (not affine — an affine defining equality
   is better handled by presolve's singleton-equality substitution) and
   the relaxed inequality is convex by the syntactic curvature walker.

Every structural analyzer here abstains *conservatively*: an
unrecognised node makes occurrence detection report "might occur" and the
affine-coefficient analyzer return ``None``. Both directions cause the
transform to skip, never to fire on an unproven model. The transform is
therefore general (it keys on structure, not on any single instance) and
can only ever leave the optimum unchanged.

References
----------
Ceccon, Siirola, Misener (2020), "SUSPECT," TOP — the "objective
  constraint" detection this mirrors.
Tawarmalani, Sahinidis (2005), "A polyhedral branch-and-cut approach to
  global optimization," Math. Prog. — BARON's epigraph handling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    IndexExpression,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# A bound is treated as "free" in the binding direction when its magnitude
# exceeds this threshold. Mirrors the solver's own large-bound handling
# (declared ±1e20 free variables in GAMS imports).
_FREE_BOUND = 1e15


def _collect_var_names(expr) -> Optional[set]:
    """Return the complete set of variable names referenced by ``expr``.

    Returns ``None`` if an unrecognised / opaque node is encountered — the
    caller must then treat the variable of interest as *possibly* present
    (the sound direction for an occurrence test).
    """
    if isinstance(expr, Variable):
        return {expr.name}
    if isinstance(expr, (Constant, Parameter)):
        return set()
    if isinstance(expr, IndexExpression):
        return _collect_var_names(expr.base)
    if isinstance(expr, UnaryOp):
        return _collect_var_names(expr.operand)
    if isinstance(expr, BinaryOp):
        left = _collect_var_names(expr.left)
        right = _collect_var_names(expr.right)
        if left is None or right is None:
            return None
        return left | right
    if isinstance(expr, FunctionCall):
        acc: set = set()
        for a in expr.args:
            names = _collect_var_names(a)
            if names is None:
                return None
            acc |= names
        return acc
    if isinstance(expr, SumExpression):
        return _collect_var_names(expr.operand)
    if isinstance(expr, SumOverExpression):
        acc = set()
        for t in expr.terms:
            names = _collect_var_names(t)
            if names is None:
                return None
            acc |= names
        return acc
    # Unknown / opaque node (CustomCall, MatMul, ...): cannot prove the
    # variable is absent — abstain.
    return None


def _occurs(expr, varname: str) -> bool:
    """True when ``varname`` *might* appear in ``expr`` (conservative)."""
    names = _collect_var_names(expr)
    if names is None:
        return True  # opaque node — assume it might occur (sound)
    return varname in names


def _affine_coeff(expr, varname: str) -> Optional[float]:
    """Constant coefficient of ``varname`` in ``expr``, or ``None``.

    Returns a float ``a`` iff ``expr`` depends on ``varname`` *only*
    affinely, i.e. ``expr == a * <varname> + (terms free of varname)`` with
    ``a`` a compile-time constant (``0.0`` means it does not appear). Returns
    ``None`` whenever the dependence is nonlinear, indexed, or routed
    through an unanalyzable node — a conservative abstention.
    """
    if isinstance(expr, Variable):
        return 1.0 if expr.name == varname else 0.0
    if isinstance(expr, (Constant, Parameter)):
        return 0.0
    if isinstance(expr, IndexExpression):
        # An indexed reference to the target variable means the target is an
        # array element; the single-free-scalar pattern does not apply.
        if _occurs(expr.base, varname):
            return None
        return 0.0
    if isinstance(expr, UnaryOp):
        inner = _affine_coeff(expr.operand, varname)
        if inner is None:
            return None
        if expr.op in ("-", "neg"):
            return -inner
        if expr.op in ("+", "pos"):
            return inner
        # abs / other unary atoms are nonlinear in their argument
        return None if _occurs(expr.operand, varname) else 0.0
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "+":
            cl = _affine_coeff(expr.left, varname)
            cr = _affine_coeff(expr.right, varname)
            if cl is None or cr is None:
                return None
            return cl + cr
        if op == "-":
            cl = _affine_coeff(expr.left, varname)
            cr = _affine_coeff(expr.right, varname)
            if cl is None or cr is None:
                return None
            return cl - cr
        if op == "*":
            lo = _occurs(expr.left, varname)
            ro = _occurs(expr.right, varname)
            if lo and ro:
                return None  # var on both sides -> nonlinear
            if not lo and not ro:
                return 0.0
            # exactly one side carries the var; the other must be a constant
            if lo:
                k = _const_value(expr.right)
                inner = _affine_coeff(expr.left, varname)
            else:
                k = _const_value(expr.left)
                inner = _affine_coeff(expr.right, varname)
            if k is None or inner is None:
                return None
            return k * inner
        if op == "/":
            # var only allowed in the numerator, denominator must be constant
            if _occurs(expr.right, varname):
                return None
            k = _const_value(expr.right)
            inner = _affine_coeff(expr.left, varname)
            if k is None or inner is None or k == 0.0:
                return None
            return inner / k
        if op == "**":
            # var under a power is nonlinear; only var-free bases are fine
            if _occurs(expr.left, varname) or _occurs(expr.right, varname):
                return None
            return 0.0
        return None if (_occurs(expr.left, varname) or _occurs(expr.right, varname)) else 0.0
    # FunctionCall / Sum / opaque: var-free -> 0, otherwise nonlinear/unknown.
    return None if _occurs(expr, varname) else 0.0


def _const_value(expr) -> Optional[float]:
    """Return a scalar constant value for ``expr`` if it is one, else ``None``."""
    if isinstance(expr, Constant):
        try:
            v = np.asarray(expr.value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if v.size == 1:
            return float(v.reshape(-1)[0])
    if isinstance(expr, Parameter):
        try:
            v = np.asarray(expr.value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if v.size == 1:
            return float(v.reshape(-1)[0])
    return None


def relax_objective_defining_equality(model):
    """Relax an objective-defining equality to its (exact) binding inequality.

    Returns ``(model, changed)``. When the structural pattern in the module
    docstring holds and the relaxed inequality is convex, returns a shallow
    copy of ``model`` with the defining equality's ``sense`` flipped to the
    binding inequality; otherwise returns ``(model, False)`` unchanged.

    The returned model never aliases the caller's constraint objects: a new
    ``Constraint`` replaces the rewritten one in a fresh ``_constraints``
    list on a shallow-copied ``Model``.
    """
    import copy as _copy

    obj = getattr(model, "_objective", None)
    if obj is None:
        return model, False
    oe = obj.expression
    # (1) objective must be exactly a single scalar variable z.
    if not isinstance(oe, Variable) or oe.size != 1:
        return model, False
    z = oe
    zname = z.name

    is_min = obj.sense == ObjectiveSense.MINIMIZE

    # (2) z must be free in the binding direction.
    try:
        z_lb = float(np.asarray(z.lb).reshape(-1)[0])
        z_ub = float(np.asarray(z.ub).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return model, False
    if is_min and z_lb > -_FREE_BOUND:
        return model, False
    if (not is_min) and z_ub < _FREE_BOUND:
        return model, False

    # (3) z appears in exactly one constraint, an equality, affinely.
    defining_idx = None
    coeff = None
    for ci, c in enumerate(model._constraints):
        if not isinstance(c, Constraint):
            # opaque constraint type (SOS / disjunction); if it might touch
            # z we cannot prove sole occurrence -> abstain.
            body = getattr(c, "body", None)
            if body is not None and _occurs(body, zname):
                return model, False
            continue
        if not _occurs(c.body, zname):
            continue
        # z occurs in this constraint.
        if defining_idx is not None:
            return model, False  # second occurrence -> not a sole-defining var
        if c.sense != "==":
            return model, False  # z occurs in a non-equality -> abstain
        a = _affine_coeff(c.body, zname)
        if a is None or a == 0.0:
            return model, False
        defining_idx = ci
        coeff = a
    if defining_idx is None:
        return model, False

    # (4) Determine the binding inequality sense and require it convex and
    #     genuinely curved (skip affine bodies — presolve substitution is
    #     better there).
    defining = model._constraints[defining_idx]
    if is_min:
        relaxed_sense = ">=" if coeff > 0 else "<="
    else:
        relaxed_sense = "<=" if coeff > 0 else ">="

    from discopt._jax.convexity import Curvature, classify_expr

    try:
        curv = classify_expr(defining.body, model)
    except Exception:
        return model, False
    if curv == Curvature.AFFINE:
        return model, False
    relaxed_is_convex = (relaxed_sense == ">=" and curv == Curvature.CONCAVE) or (
        relaxed_sense == "<=" and curv == Curvature.CONVEX
    )
    if not relaxed_is_convex:
        return model, False

    # Build the rewritten model without mutating the caller's objects.
    new_constraints = list(model._constraints)
    new_constraints[defining_idx] = Constraint(
        body=defining.body,
        sense=relaxed_sense,
        rhs=defining.rhs,
        name=defining.name,
    )
    new_model = _copy.copy(model)
    new_model._constraints = new_constraints
    return new_model, True


__all__ = ["relax_objective_defining_equality"]
