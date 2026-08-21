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

from typing import Any, Dict, Optional, Tuple

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


_EMPTY_NAMES: frozenset = frozenset()


def _name_walk_children(expr):
    """``(kind, children)`` for the variable-name walk.

    ``kind`` is ``"var"`` (a scalar variable reference), ``"empty"`` (provably
    variable-free), ``"combine"`` (the name set is the union of the children's,
    ``None`` if any child is ``None``), or ``"opaque"`` (unrecognised node —
    abstain with ``None``).

    The isinstance ladder must stay in the same order as the semantics it
    encodes; :func:`_collect_var_names` and :class:`VarNameIndex` both read it,
    so the two can never drift apart.
    """
    if isinstance(expr, Variable):
        return "var", ()
    if isinstance(expr, (Constant, Parameter)):
        return "empty", ()
    if isinstance(expr, IndexExpression):
        return "combine", (expr.base,)
    if isinstance(expr, UnaryOp):
        return "combine", (expr.operand,)
    if isinstance(expr, BinaryOp):
        return "combine", (expr.left, expr.right)
    if isinstance(expr, FunctionCall):
        return "combine", tuple(expr.args)
    if isinstance(expr, SumExpression):
        return "combine", (expr.operand,)
    if isinstance(expr, SumOverExpression):
        return "combine", tuple(expr.terms)
    # Unknown / opaque node (CustomCall, MatMul, ...): cannot prove the
    # variable is absent — abstain.
    return "opaque", ()


class VarNameIndex:
    """Memoized variable-name sets for every node of one expression DAG.

    Built once in ``O(nodes)``; :meth:`names` and :meth:`occurs` are then ``O(1)``
    dict lookups. The naive recursive walk allocates a fresh set at every node
    and re-expands shared subexpressions, so an occurrence test called at each
    level of a structural analyzer costs ``O(nodes^2)`` per variable — the
    ``t1000`` non-termination in issue #1104 (1002 candidate names against a
    5003-node, depth-1004 sum chain).

    Two properties matter for correctness:

    * **DAG-correct.** Keyed by ``id(node)``, so a subexpression reachable
      through several parents is walked once. The index holds a reference to
      every node it memoizes (values are ``(node, names)``), so CPython can
      never recycle an ``id`` while the index is alive — the usual hazard of
      ``id()``-keyed memos.
    * **Iterative.** An explicit stack, not recursion: a depth-1004 body is
      already close to the default recursion limit, and MINLPLib carries
      deeper ones.

    An index is valid only for expressions that are *not mutated* while it
    lives; the analyzers here build one per constraint body and discard it.
    """

    __slots__ = ("_memo",)

    _memo: Dict[int, Tuple[Any, Optional[frozenset]]]

    def __init__(self, root):
        # id(node) -> (node, names|None). The node is stored to pin the id.
        memo: Dict[int, Tuple[Any, Optional[frozenset]]] = {}
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            key = id(node)
            if key in memo:
                continue
            kind, children = _name_walk_children(node)
            if not expanded and children:
                # Post-order: revisit this node once every child is memoized.
                stack.append((node, True))
                for child in children:
                    if id(child) not in memo:
                        stack.append((child, False))
                continue
            if kind == "var":
                memo[key] = (node, frozenset((node.name,)))
                continue
            if kind == "empty":
                memo[key] = (node, _EMPTY_NAMES)
                continue
            if kind == "opaque":
                memo[key] = (node, None)
                continue
            # "combine" with no children collapses to the empty set, matching
            # the union-over-nothing of the recursive walk.
            acc: Optional[set] = set()
            for child in children:
                sub = memo[id(child)][1]
                if sub is None:
                    acc = None
                    break
                acc |= sub
            memo[key] = (node, None if acc is None else frozenset(acc))
        self._memo = memo

    def __len__(self) -> int:
        """Number of distinct DAG nodes indexed."""
        return len(self._memo)

    def names(self, expr) -> Optional[frozenset]:
        """Variable names referenced by ``expr``, or ``None`` if opaque.

        ``expr`` must be a node of the DAG this index was built from; a foreign
        node raises ``KeyError`` rather than silently reporting "no variables"
        (a wrong-but-plausible answer that would make an occurrence test unsound).
        """
        return self._memo[id(expr)][1]

    def occurs(self, expr, varname: str) -> bool:
        """True when ``varname`` *might* appear in ``expr`` (conservative)."""
        names = self._memo[id(expr)][1]
        if names is None:
            return True  # opaque node — assume it might occur (sound)
        return varname in names


def _collect_var_names(expr) -> Optional[set]:
    """Return the complete set of variable names referenced by ``expr``.

    Returns ``None`` if an unrecognised / opaque node is encountered — the
    caller must then treat the variable of interest as *possibly* present
    (the sound direction for an occurrence test).

    Returns a fresh mutable ``set`` the caller may modify. When several
    occurrence queries are made against one expression, build a
    :class:`VarNameIndex` once instead of calling this repeatedly.
    """
    names = VarNameIndex(expr).names(expr)
    return None if names is None else set(names)


def _occurs(expr, varname: str) -> bool:
    """True when ``varname`` *might* appear in ``expr`` (conservative)."""
    return VarNameIndex(expr).occurs(expr, varname)


def _affine_coeff(expr, varname: str, index: Optional[VarNameIndex] = None) -> Optional[float]:
    """Constant coefficient of ``varname`` in ``expr``, or ``None``.

    Returns a float ``a`` iff ``expr`` depends on ``varname`` *only*
    affinely, i.e. ``expr == a * <varname> + (terms free of varname)`` with
    ``a`` a compile-time constant (``0.0`` means it does not appear). Returns
    ``None`` whenever the dependence is nonlinear, indexed, or routed
    through an unanalyzable node — a conservative abstention.

    ``index`` is the memoized occurrence index for ``expr``'s DAG; it is built
    on entry when omitted. Passing it (or letting the recursion thread it, as
    it does here) keeps the walk ``O(nodes)`` instead of ``O(nodes^2)`` — see
    :class:`VarNameIndex`.
    """
    if index is None:
        index = VarNameIndex(expr)
    if isinstance(expr, Variable):
        return 1.0 if expr.name == varname else 0.0
    if isinstance(expr, (Constant, Parameter)):
        return 0.0
    if isinstance(expr, IndexExpression):
        # An indexed reference to the target variable means the target is an
        # array element; the single-free-scalar pattern does not apply.
        if index.occurs(expr.base, varname):
            return None
        return 0.0
    if isinstance(expr, UnaryOp):
        inner = _affine_coeff(expr.operand, varname, index)
        if inner is None:
            return None
        if expr.op in ("-", "neg"):
            return -inner
        if expr.op in ("+", "pos"):
            return inner
        # abs / other unary atoms are nonlinear in their argument
        return None if index.occurs(expr.operand, varname) else 0.0
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "+":
            cl = _affine_coeff(expr.left, varname, index)
            cr = _affine_coeff(expr.right, varname, index)
            if cl is None or cr is None:
                return None
            return cl + cr
        if op == "-":
            cl = _affine_coeff(expr.left, varname, index)
            cr = _affine_coeff(expr.right, varname, index)
            if cl is None or cr is None:
                return None
            return cl - cr
        if op == "*":
            lo = index.occurs(expr.left, varname)
            ro = index.occurs(expr.right, varname)
            if lo and ro:
                return None  # var on both sides -> nonlinear
            if not lo and not ro:
                return 0.0
            # exactly one side carries the var; the other must be a constant
            if lo:
                k = _const_value(expr.right)
                inner = _affine_coeff(expr.left, varname, index)
            else:
                k = _const_value(expr.left)
                inner = _affine_coeff(expr.right, varname, index)
            if k is None or inner is None:
                return None
            return k * inner
        if op == "/":
            # var only allowed in the numerator, denominator must be constant
            if index.occurs(expr.right, varname):
                return None
            k = _const_value(expr.right)
            inner = _affine_coeff(expr.left, varname, index)
            if k is None or inner is None or k == 0.0:
                return None
            return inner / k
        if op == "**":
            # var under a power is nonlinear; only var-free bases are fine
            if index.occurs(expr.left, varname) or index.occurs(expr.right, varname):
                return None
            return 0.0
        if index.occurs(expr.left, varname) or index.occurs(expr.right, varname):
            return None
        return 0.0
    # FunctionCall / Sum / opaque: var-free -> 0, otherwise nonlinear/unknown.
    return None if index.occurs(expr, varname) else 0.0


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

    from discopt._relax.convexity import Curvature, classify_expr

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
