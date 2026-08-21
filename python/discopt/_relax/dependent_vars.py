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
# objective-defining-equality relaxation. ``VarNameIndex`` reports ``None`` for
# an opaque node, which every query here reads as "might occur" (the safe
# direction). We do NOT reuse that module's ``_affine_coeff``: it is written for
# the narrower contract "the whole body is affine in z", so a sibling term that
# is nonlinear in *other* variables (e.g. ``4243.28/(x0*x1)``) makes it return
# None even when the target variable itself appears purely affinely. The
# isolating extractor below short-circuits on "target absent -> coefficient 0",
# which is what the dependent-output pattern requires.
from discopt._relax.objective_epigraph import VarNameIndex, WorkCounter, _const_value
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

# Traversal cap for :func:`_isolated_affine_coeffs`, in node visits per distinct
# DAG node. A tree needs exactly one visit per node; sharing multiplies that by
# the number of distinct paths. Measured over 149,564 equality bodies from 104
# MINLPLib instances (issue #1104) the worst ratio was 1.0 — the parsed bodies
# are trees — so this leaves three orders of magnitude of headroom before the
# conservative abstention can fire.
_VISIT_BUDGET_PER_NODE = 1000
_VISIT_BUDGET_FLOOR = 10_000

# Deterministic work allowance for one whole call to
# :func:`find_functionally_dependent_names`, in elementary operations (node
# visits plus name-set element unions; see :class:`WorkCounter`).
#
# Why a work count and not a wall deadline: this scan runs *before* the B&B loop
# arms ``time_limit``, so it must be bounded by something — but the set it
# returns steers spatial branching, so bounding it with a clock would make the
# search tree a function of machine speed, the exact defect #912 exists to
# prevent (see ``python/tests/test_912_wall_budget_inventory.py``). A work count
# stops the overrun of issue #1104 while keeping ``deterministic=True`` solves
# reproducible.
#
# Calibration (issue #1104), measured over 406 MINLPLib instances that reach the
# scan (the other 46 of the 452-instance corpus have no scalar continuous
# candidate and cost nothing): median demand 805 units, p95 288k, p99 4.6M, max
# 56.2M (``torsion100``). Observed throughput 2.8e6-3.4e7 units/s, so 20M bounds
# the whole scan at ~7 s even at the slowest per-unit rate observed, and at ~0.6 s
# on the instance that actually reaches the cap.
#
# At 20M exactly ONE of the 406 truncates -- ``torsion100``, which drops from 2
# detected names to 0. That is a subset (verified, along with reproducibility, on
# the 20 costliest instances), and ``torsion100`` is one of the three instances
# the pre-#1104 scan could not finish at all. The runner-up, ``junkturn`` at
# 9.2M, keeps 2.2x headroom.
_SCAN_WORK_BUDGET = 20_000_000


def _isolated_affine_coeffs(expr, index=None, work=None):
    """Isolated affine coefficients of *every* variable in ``expr``, in one pass.

    Returns a mapping ``name -> coeff`` where ``coeff`` is a float when ``name``
    occurs affinely with a constant coefficient, and ``None`` when it occurs
    nonlinearly (a product carrying it twice, under a power, a nonlinear
    unary/function, in a denominator, indexed, or inside a reduction). A name
    absent from the mapping is provably absent from ``expr`` (coefficient 0).
    Returns ``None`` — "no name can be proven affine here" — when ``expr``
    contains an opaque node, since occurrence detection must then report "might
    occur" for every name.

    Unlike a whole-expression affine test, absence short-circuits: a
    subexpression not containing a name contributes coefficient ``0`` to it
    regardless of its own nonlinearity. This is what lets a defining equality
    like ``x6 - sqrt(g(others)) == 0`` report ``coeff[x6] = 1`` even though ``g``
    is highly nonlinear in the other variables.

    Cost is ``O(nodes)`` on a tree: one top-down traversal carrying the
    accumulated constant multiplier, with ``O(1)`` occurrence tests from
    ``index``. The per-name formulation it replaces re-walked the body for each
    candidate and re-derived occurrence at every level, which is
    ``O(names x nodes^2)`` — the ``t1000`` non-termination of issue #1104. The
    traversal is iterative because MINLPLib sum chains are deeper than the
    default recursion limit.

    The multiplier a node inherits depends on the path taken to reach it, so a
    *shared* subexpression is visited once per path and the walk is O(paths),
    not O(nodes). ``_VISIT_BUDGET_PER_NODE`` caps that at a fixed multiple of the
    DAG size and abstains (returns ``None``) beyond it, so a pathologically
    shared body degrades to "nothing proven here" instead of to a hang. The cap
    is on a node counter, not a clock: the detected set must not depend on how
    fast the machine is (``deterministic=True`` solves reproduce).
    """
    if index is None:
        index = VarNameIndex(expr, work=work)
    if index.names(expr) is None:
        return None  # opaque node reachable -> nothing provable (see docstring)

    coeffs: dict = {}
    bad: set = set()
    budget = _VISIT_BUDGET_PER_NODE * len(index) + _VISIT_BUDGET_FLOOR
    stack = [(expr, 1.0)]
    while stack:
        budget -= 1
        if budget < 0:
            return None  # shared-subexpression blowup -> abstain (see docstring)
        if work is not None and not work.spend(1):
            return None  # scan-wide work allowance gone -> abstain (see docstring)
        node, mult = stack.pop()
        names = index.names(node)
        if not names:
            continue  # variable-free subtree -> contributes 0 to every name
        if isinstance(node, Variable):
            coeffs[node.name] = coeffs.get(node.name, 0.0) + mult
            continue
        if isinstance(node, IndexExpression):
            bad |= names  # indexed occurrence -> not the scalar affine pattern
            continue
        if isinstance(node, UnaryOp):
            if node.op in ("-", "neg"):
                stack.append((node.operand, -mult))
            elif node.op in ("+", "pos"):
                stack.append((node.operand, mult))
            else:
                bad |= names  # abs / sin / exp / ... over a variable-carrying arg
            continue
        if isinstance(node, BinaryOp):
            op = node.op
            if op == "+":
                stack.append((node.left, mult))
                stack.append((node.right, mult))
                continue
            if op == "-":
                stack.append((node.left, mult))
                stack.append((node.right, -mult))
                continue
            if op == "*":
                left_names = index.names(node.left)
                right_names = index.names(node.right)
                if left_names and right_names:
                    bad |= names  # variables on both factors -> nonlinear
                    continue
                # Exactly one factor carries variables; the other must be a
                # constant for any name under it to stay affine.
                if left_names:
                    k = _const_value(node.right)
                    carrier = node.left
                else:
                    k = _const_value(node.left)
                    carrier = node.right
                if k is None:
                    bad |= names
                else:
                    stack.append((carrier, mult * k))
                continue
            if op == "/":
                if index.names(node.right):
                    bad |= names  # variable in a denominator -> nonlinear
                    continue
                k = _const_value(node.right)
                if k is None or k == 0.0:
                    bad |= names
                else:
                    stack.append((node.left, mult / k))
                continue
            # ``**`` (variable under a power) and unrecognised binary ops.
            bad |= names
            continue
        if isinstance(node, SumExpression):
            stack.append((node.operand, mult))
            continue
        # SumOverExpression (a reduction carrying the name -> not a scalar
        # coefficient), FunctionCall, or any other node over variables.
        bad |= names

    for name in bad:
        coeffs[name] = None
    return coeffs


def _isolated_affine_coeff(expr, varname, index=None):
    """Constant coefficient of ``varname`` in ``expr`` if it occurs *affinely*.

    Returns a float ``a`` iff ``expr == a*<varname> + (terms not involving
    varname)``, where those other terms may be arbitrarily nonlinear in *other*
    variables; ``None`` when ``varname`` occurs nonlinearly or unanalyzably, and
    ``0.0`` when it is provably absent.

    Thin single-name view of :func:`_isolated_affine_coeffs`; callers with more
    than one name to test should use that directly and pay the traversal once.
    """
    coeffs = _isolated_affine_coeffs(expr, index)
    if coeffs is None:
        return None
    return coeffs.get(varname, 0.0)


def _carries(index, node) -> bool:
    """True when ``node`` references (or might reference) any variable."""
    names = index.names(node)
    if names is None:
        return True  # opaque node — assume it could carry a variable (sound)
    return bool(names)


def _carries_variable(expr) -> bool:
    """True when ``expr`` references (or might reference) any variable."""
    return _carries(VarNameIndex(expr), expr)


def _body_is_nonlinear(expr, index=None) -> bool:
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

    ``index`` supplies ``O(1)`` variable-carrying tests (built on entry when
    omitted); the traversal is iterative so a deep sum chain cannot exhaust the
    recursion limit.
    """
    if index is None:
        index = VarNameIndex(expr)
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, (Variable, Constant, Parameter)):
            continue
        if isinstance(node, UnaryOp):
            if node.op in ("-", "neg", "+", "pos"):
                stack.append(node.operand)
                continue
            # abs / sin / cos / exp / log ... are nonlinear over a variable argument
            if _carries(index, node.operand):
                return True
            continue
        if isinstance(node, BinaryOp):
            op = node.op
            if op in ("+", "-"):
                stack.append(node.left)
                stack.append(node.right)
                continue
            if op == "*":
                if _carries(index, node.left) and _carries(index, node.right):
                    return True
                stack.append(node.left)
                stack.append(node.right)
                continue
            if op == "/":
                if _carries(index, node.right):
                    return True
                stack.append(node.left)
                continue
            if op == "**":
                if _carries(index, node.left) or _carries(index, node.right):
                    return True
                continue
            # Unknown binary op over variables -> treat as nonlinear (conservative).
            if _carries(index, node.left) or _carries(index, node.right):
                return True
            continue
        if isinstance(node, SumExpression):
            stack.append(node.operand)
            continue
        if isinstance(node, SumOverExpression):
            stack.extend(node.terms)
            continue
        if isinstance(node, FunctionCall):
            if any(_carries(index, a) for a in node.args):
                return True
            continue
        # Opaque / unrecognised node referencing variables -> nonlinear (sound).
        if _carries(index, node):
            return True
    return False


def find_functionally_dependent_names(model, work_budget: int | None = None) -> set:
    """Names of continuous scalar variables pinned by a nonlinear equality.

    Returns the set of variable names ``x`` such that some ``==`` constraint has
    a *nonlinear* body in which ``x`` appears affinely with a constant nonzero
    coefficient. Such ``x`` is determined as a function of the other variables;
    spatial branching should deprioritize it (see module docstring).

    Only scalar (size-1) continuous variables are considered: the affine
    analyzer abstains on indexed/array references, and the single-defining-
    equality pattern is per-scalar.

    ``work_budget`` caps the whole scan in elementary operations (default
    :data:`_SCAN_WORK_BUDGET`); pass ``0`` to skip the scan outright. When it
    runs out the scan stops and returns what it has found so far. A budget is
    needed because this runs *before* the branch-and-bound loop arms
    ``time_limit``, so an unbounded pass on a pathologically large model can
    overrun the user's limit before a single node is explored (issue #1104).

    The cap is deterministic — an operation count, not a clock — because the
    returned set steers spatial branching: a wall deadline would make the search
    tree a function of machine speed (#912). Stopping early is sound either way:
    the result feeds a deprioritization with a completeness-preserving fallback,
    so a partial set only changes branch order (the module docstring's soundness
    argument holds for *any* subset, including the empty one).
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

    work = WorkCounter(_SCAN_WORK_BUDGET if work_budget is None else work_budget)

    dependent: set = set()
    for c in getattr(model, "_constraints", []):
        if not isinstance(c, Constraint):
            continue
        if c.sense != "==":
            continue
        if work.exhausted:
            # Out of budget: return the (sound) partial set rather than overrun.
            break
        body = c.body
        # One memoized occurrence index per body serves every query below —
        # presence, nonlinearity, and the affine coefficients — so the whole
        # constraint costs O(nodes) instead of O(candidates x nodes^2). The
        # index build charges the dominant cost (one unit per node plus one per
        # unioned name), which also bounds the two O(nodes) walks below.
        index = VarNameIndex(body, work=work)
        # Names actually present in this equality, restricted to candidates not
        # already marked. ``index.names`` is None on an opaque body; skip — we
        # cannot isolate any variable affinely there.
        present = index.names(body)
        if present is None:
            continue
        names = (present & candidates) - dependent
        if not names:
            continue
        # Only a genuinely nonlinear defining equality makes branching on the
        # pinned output wasteful; affine ones are presolve's job.
        if not _body_is_nonlinear(body, index):
            continue
        coeffs = _isolated_affine_coeffs(body, index, work=work)
        if coeffs is None:
            continue
        for name in names:
            a = coeffs.get(name, 0.0)
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
