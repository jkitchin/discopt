"""Detect functionally-dependent continuous variables for spatial branching.

A continuous variable is *functionally dependent* when some equality constraint
pins it as a function of the other variables, i.e.

    x_i = f(other variables)          (a defining equality)

structurally recognised as: ``x_i`` appears in an equality constraint
*affinely* with a constant nonzero coefficient, so the equality
``a*x_i + g(rest) == rhs`` can be solved for ``x_i = (rhs - g(rest))/a``.

Why spatial branch-and-bound should deprioritize these
------------------------------------------------------
In nonconvex spatial B&B the convex (McCormick / factorable) relaxation gap is
driven by the *independent* inputs of each nonlinear term. A dependent variable
is an *output*: once the independent inputs of its defining equality are fixed,
its value is determined, and bound tightening (FBBT/OBBT) recovers it for free.
Bisecting a dependent output therefore spends a branch without shrinking the
relaxation gap — the gap lives on the inputs. Branching should target the
independent drivers first.

This is the generic mechanism behind the welded-beam (nvs05) certification: its
stress intermediates (``x5..x8`` — shear/bending/buckling terms) are each
defined by a single nonlinear equality, while only ``{i1,i2,x3,x4}`` actually
drive the objective and the McCormick gap. Branching solely on the independent
variables certifies the global optimum in ~23 nodes instead of stalling.

Soundness
---------
The result feeds a *deprioritization with fallback* in the Rust spatial
selector (``select_spatial_branch_variable``): dependent variables are branched
only when no independent continuous variable still qualifies. This changes
branch *order* only — never the relaxation, the bounds, or whether a branchable
dimension is refused — so completeness and soundness of the search are
preserved regardless of how aggressively this detector marks variables. The
detector keys on model structure (not on any single instance) and abstains
conservatively: an unrecognised node makes occurrence detection report "might
occur" and the affine-coefficient analyzer return ``None``, both of which cause
the variable to be left *un*marked (independent) rather than wrongly skipped.

Detection is intended to run on the *original* model, before the factorable
reformulation rewrites a defining equality like ``x5 = c/(x3*x4)`` into a
product form ``x5*x3*x4 == c`` (in which ``x5`` is no longer affine). The
functional-dependency property is invariant under that rewrite — ``x5`` is
still determined by ``x3,x4`` — so the names captured on the original model
remain the correct ones to deprioritize in the lifted/solved model.
"""

from __future__ import annotations

import numpy as np

# Reuse the conservative occurrence/constant analyzers proven in the
# objective-defining-equality relaxation. ``_occurs`` reports "might occur" on
# any opaque node (the safe direction), and ``_collect_var_names`` returns None
# there. We do NOT reuse that module's ``_affine_coeff``: it is written for the
# narrower contract "the whole body is affine in z", so a sibling term that is
# nonlinear in *other* variables (e.g. ``4243.28/(x0*x1)``) makes it return
# None even when the target variable itself appears purely affinely. The
# isolating extractor below short-circuits on "target absent -> coefficient 0",
# which is what the dependent-output pattern requires.
from discopt._jax.objective_epigraph import _collect_var_names, _const_value, _occurs
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    IndexExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


def _isolated_affine_coeff(expr, varname):
    """Constant coefficient of ``varname`` in ``expr`` if it occurs *affinely*.

    Returns a float ``a`` iff ``expr == a*<varname> + (terms not involving
    varname)``, where those other terms may be arbitrarily nonlinear in *other*
    variables. Returns ``None`` when ``varname`` occurs nonlinearly (a product
    carrying it twice, under a power, a nonlinear unary/function, in a
    denominator, indexed, a reduction, or an opaque node) and ``0.0`` when it is
    provably absent.

    Unlike a whole-expression affine test, every node first short-circuits on
    absence: a subexpression not containing ``varname`` contributes coefficient
    ``0`` regardless of its own nonlinearity. This is what lets a defining
    equality like ``x6 - sqrt(g(others)) == 0`` report ``coeff[x6] = 1`` even
    though ``g`` is highly nonlinear in the other variables.
    """
    # Absent (and not opaque) -> contributes nothing. ``_occurs`` returns True
    # on opaque nodes, so reaching past here means varname genuinely occurs.
    if not _occurs(expr, varname):
        return 0.0
    if isinstance(expr, Variable):
        return 1.0 if expr.name == varname else 0.0
    if isinstance(expr, IndexExpression):
        return None  # indexed occurrence -> not the scalar affine pattern
    if isinstance(expr, UnaryOp):
        if expr.op in ("-", "neg"):
            inner = _isolated_affine_coeff(expr.operand, varname)
            return None if inner is None else -inner
        if expr.op in ("+", "pos"):
            return _isolated_affine_coeff(expr.operand, varname)
        return None  # abs / sin / exp / ... applied to a varname-carrying arg
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op in ("+", "-"):
            cl = _isolated_affine_coeff(expr.left, varname)
            cr = _isolated_affine_coeff(expr.right, varname)
            if cl is None or cr is None:
                return None
            return cl - cr if op == "-" else cl + cr
        if op == "*":
            lo = _occurs(expr.left, varname)
            ro = _occurs(expr.right, varname)
            if lo and ro:
                return None  # varname on both factors -> nonlinear
            # Exactly one factor carries varname; the other must be constant.
            if lo:
                k = _const_value(expr.right)
                inner = _isolated_affine_coeff(expr.left, varname)
            else:
                k = _const_value(expr.left)
                inner = _isolated_affine_coeff(expr.right, varname)
            if k is None or inner is None:
                return None
            return k * inner
        if op == "/":
            if _occurs(expr.right, varname):
                return None  # varname in a denominator -> nonlinear
            k = _const_value(expr.right)
            inner = _isolated_affine_coeff(expr.left, varname)
            if k is None or inner is None or k == 0.0:
                return None
            return inner / k
        if op == "**":
            return None  # varname occurs under a power -> nonlinear
        return None
    if isinstance(expr, SumExpression):
        return _isolated_affine_coeff(expr.operand, varname)
    if isinstance(expr, SumOverExpression):
        return None  # reduction carrying varname -> not a scalar coefficient
    # FunctionCall / opaque node carrying varname -> nonlinear.
    return None


def _carries_variable(expr) -> bool:
    """True when ``expr`` references (or might reference) any variable."""
    names = _collect_var_names(expr)
    if names is None:
        return True  # opaque node — assume it could carry a variable (sound)
    return len(names) > 0


def _body_is_nonlinear(expr) -> bool:
    """Cheap structural test: does ``expr`` contain genuine nonlinearity?

    A *nonlinear* node is a product of two variable-carrying factors, a division
    by a variable-carrying denominator, a power with a variable base/exponent, a
    nonlinear unary/function applied to a variable-carrying argument, or any
    opaque node over variables. Pure affine combinations return ``False``.

    Used only to gate deprioritization to variables pinned by a *nonlinear*
    defining equality (where branching on the output is actually wasteful);
    affine defining equalities are handled by presolve singleton substitution
    and branch order on them is immaterial. Conservative: an unrecognised node
    counts as nonlinear, which can only *add* a variable to the deprioritized
    set — safe under the fallback.
    """
    if isinstance(expr, (Variable, Constant, Parameter)):
        return False
    if isinstance(expr, UnaryOp):
        if expr.op in ("-", "neg", "+", "pos"):
            return _body_is_nonlinear(expr.operand)
        # abs / sin / cos / exp / log ... are nonlinear over a variable argument
        return _carries_variable(expr.operand)
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op in ("+", "-"):
            return _body_is_nonlinear(expr.left) or _body_is_nonlinear(expr.right)
        if op == "*":
            if _carries_variable(expr.left) and _carries_variable(expr.right):
                return True
            return _body_is_nonlinear(expr.left) or _body_is_nonlinear(expr.right)
        if op == "/":
            if _carries_variable(expr.right):
                return True
            return _body_is_nonlinear(expr.left)
        if op == "**":
            if _carries_variable(expr.left) or _carries_variable(expr.right):
                return True
            return False
        # Unknown binary op over variables -> treat as nonlinear (conservative).
        return _carries_variable(expr.left) or _carries_variable(expr.right)
    if isinstance(expr, SumExpression):
        return _body_is_nonlinear(expr.operand)
    if isinstance(expr, SumOverExpression):
        return any(_body_is_nonlinear(t) for t in expr.terms)
    if isinstance(expr, FunctionCall):
        return any(_carries_variable(a) for a in expr.args)
    # Opaque / unrecognised node referencing variables -> nonlinear (sound).
    return _carries_variable(expr)


def find_functionally_dependent_names(model) -> set:
    """Names of continuous scalar variables pinned by a nonlinear equality.

    Returns the set of variable names ``x`` such that some ``==`` constraint has
    a *nonlinear* body in which ``x`` appears affinely with a constant nonzero
    coefficient. Such ``x`` is determined as a function of the other variables;
    spatial branching should deprioritize it (see module docstring).

    Only scalar (size-1) continuous variables are considered: the affine
    analyzer abstains on indexed/array references, and the single-defining-
    equality pattern is per-scalar.
    """
    # Candidate names: scalar continuous variables only.
    candidates: set = set()
    for v in getattr(model, "_variables", []):
        if getattr(v, "var_type", None) != VarType.CONTINUOUS:
            continue
        if getattr(v, "size", 1) != 1:
            continue
        candidates.add(v.name)
    if not candidates:
        return set()

    dependent: set = set()
    for c in getattr(model, "_constraints", []):
        if not isinstance(c, Constraint):
            continue
        if c.sense != "==":
            continue
        body = c.body
        # Names actually present in this equality, restricted to candidates not
        # already marked. ``_collect_var_names`` returns None on an opaque body;
        # skip — we cannot isolate any variable affinely there.
        present = _collect_var_names(body)
        if present is None:
            continue
        names = (present & candidates) - dependent
        if not names:
            continue
        # Only a genuinely nonlinear defining equality makes branching on the
        # pinned output wasteful; affine ones are presolve's job.
        if not _body_is_nonlinear(body):
            continue
        for name in names:
            a = _isolated_affine_coeff(body, name)
            if a is None or a == 0.0 or not np.isfinite(a):
                continue
            dependent.add(name)
    return dependent


def dependent_columns_for_model(model, names: set) -> list:
    """Flat column indices in ``model`` of the variables in ``names``.

    Maps the dependent variable *names* (typically detected on the pre-reform
    model) onto the flat columns of ``model`` (typically the lifted/solved
    model). Only continuous columns are emitted — integer variables are handled
    by integer branching and are skipped by the spatial selector anyway.
    """
    if not names:
        return []
    cols: list = []
    off = 0
    for v in getattr(model, "_variables", []):
        size = int(getattr(v, "size", 1))
        if getattr(v, "var_type", None) == VarType.CONTINUOUS and v.name in names and size == 1:
            cols.append(off)
        off += size
    return cols


__all__ = [
    "find_functionally_dependent_names",
    "dependent_columns_for_model",
]
