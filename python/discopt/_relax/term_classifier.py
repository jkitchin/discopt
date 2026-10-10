"""
Nonlinear term classifier.

Walks the expression DAG of a Model and catalogs nonlinear term structure:
  - bilinear terms:   x_i * x_j  (two distinct continuous variables)
  - trilinear terms:  x_i * x_j * x_k  (three distinct continuous variables)
  - multilinear terms: x_i * ... * x_k (four or more distinct variables)
  - monomial terms:   x_i^n  (single variable raised to integer power n ≥ 2)
  - general_nl:       all other nonlinearities (sin, cos, exp, log, etc.)

This catalog drives:
  1. Variable selection for partitioning (which variables appear in nonlinear terms)
  2. MILP relaxation construction (which terms get McCormick / lambda constraints)
  3. Interaction graph for min-vertex-cover variable selection

It originated as AMP infrastructure and is named for that in the theory references
below, but it is **general**: every nonlinear route consults
:func:`classify_nonlinear_terms`, which prefers the Rust arena classifier
(``discopt-core::term_classifier``, called ``amp.rs`` until #1343) and falls back to
the Python walk here. Which of the two ran, and why, is recorded — see
:func:`classifier_route_counts`.

Theory references:
  - Nagarajan et al., CP 2016: http://harshangrjn.github.io/pdf/CP_2016.pdf
  - Nagarajan et al., JOGO 2018: http://harshangrjn.github.io/pdf/JOGO_2018.pdf
  - Alpine.jl operators.jl / nlexpr.jl
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from discopt._flat_index import resolve_scalar_slot
from discopt._relax.scalarize import scalar_elements, scalar_matmul_contraction, static_shape
from discopt._work_budget import WorkBudget
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Which classifier ran, and why (issue #1343)
# ---------------------------------------------------------------------------
#
# :func:`classify_nonlinear_terms` prefers the Rust arena classifier and falls
# back to the Python walk. Until #1343 that fallback was silent — three bare
# ``except Exception: return None`` arms plus three structural declines, all
# indistinguishable from a Rust success, so "the Rust path is in use" was an
# assumption rather than a measured fact (CLAUDE.md §7 applied to production
# code). Every route is now named, counted and logged.
#
# The counters are a lock-guarded module global rather than a ``ContextVar`` on
# purpose. ``_run_with_deep_recursion`` runs the classification on a worker
# thread inside a *copy* of the caller's context, and its own docstring states
# that "writes made by ``fn`` still cannot leak back here" — so a ContextVar
# written during classification would record nothing on exactly the deep models
# the fallback matters most for, while working fine on small ones. That is an
# instrument that silently measures nothing (CLAUDE.md §6). Recording happens on
# the caller's thread instead, after the runner returns.

#: The Rust arena classifier produced the catalog.
ROUTE_RUST = "rust"
#: Model has an expandable square; the Rust path is skipped before it is tried.
ROUTE_PY_EXPANDABLE_SQUARE = "python:expandable_square"
#: ``discopt._rust`` could not be imported (extension not built).
ROUTE_PY_IMPORT_FAILED = "python:rust_import_failed"
#: ``model_to_repr``/``classify_nonlinear_terms`` raised.
ROUTE_PY_CLASSIFY_RAISED = "python:rust_classify_raised"
#: Payload reports ``general_nl`` terms, whose public API needs Python objects.
ROUTE_PY_GENERAL_NL_OBJECTS = "python:general_nl_expression_objects_needed"
#: The degree cross-check (``is_objective_linear``/``is_constraint_linear``) raised.
ROUTE_PY_DEGREE_CHECK_RAISED = "python:degree_check_raised"
#: Model is provably nonlinear but the payload catalogued nothing.
ROUTE_PY_CATALOG_INCOMPLETE = "python:rust_catalog_incomplete"

_ROUTE_LOCK = threading.Lock()
_ROUTE_COUNTS: dict[str, int] = {}
_ROUTE_LAST_DETAIL: dict[str, str] = {}


@dataclass(frozen=True)
class _RustAttempt:
    """Outcome of the Rust fast path: the catalog, or the route that declined it."""

    terms: "NonlinearTerms | None"
    route: str
    detail: str | None = None


def classifier_route_counts() -> dict[str, int]:
    """Return ``{route: times taken}`` since the last reset, as a copy.

    Keys are the ``ROUTE_*`` constants in this module. ``ROUTE_RUST`` counts the
    solves where the Rust classifier produced the catalog; every other key is a
    fallback to the Python walk and names its reason.
    """
    with _ROUTE_LOCK:
        return dict(_ROUTE_COUNTS)


def classifier_route_details() -> dict[str, str]:
    """Return ``{route: last detail}`` — the most recent exception repr per route.

    Populated only for the routes that decline because something raised; the
    structural declines carry no detail.
    """
    with _ROUTE_LOCK:
        return dict(_ROUTE_LAST_DETAIL)


def reset_classifier_route_counts() -> None:
    """Clear the route counters and details. Intended for tests and probes."""
    with _ROUTE_LOCK:
        _ROUTE_COUNTS.clear()
        _ROUTE_LAST_DETAIL.clear()


def _record_route(attempt: _RustAttempt) -> None:
    """Count ``attempt``'s route and log it. Called on the CALLER's thread."""
    with _ROUTE_LOCK:
        _ROUTE_COUNTS[attempt.route] = _ROUTE_COUNTS.get(attempt.route, 0) + 1
        if attempt.detail is not None:
            _ROUTE_LAST_DETAIL[attempt.route] = attempt.detail

    if attempt.route == ROUTE_RUST:
        logger.debug("term classifier: Rust arena classifier produced the catalog")
        return
    if attempt.detail is not None and attempt.route != ROUTE_PY_IMPORT_FAILED:
        # Something raised inside the Rust path. That is a defect in it, not a
        # routine decline, so it is not whispered.
        logger.warning(
            "term classifier fell back to the Python walk: %s (%s)",
            attempt.route,
            attempt.detail,
        )
        return
    logger.debug("term classifier fell back to the Python walk: %s", attempt.route)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

# Flat variable index type alias for clarity
_VarIdx = int


@dataclass
class NonlinearTerms:
    """Catalog of nonlinear term structure for AMP.

    Attributes
    ----------
    bilinear : list of (int, int)
        Each entry is a pair of flat variable indices (i, j) for a term x_i * x_j.
        The pair is always sorted (i <= j) to avoid duplicates.
    trilinear : list of (int, int, int)
        Each entry is a sorted triple of flat variable indices for x_i * x_j * x_k.
    multilinear : list of tuple[int, ...]
        Each entry is a sorted tuple of four or more flat variable indices for
        a distinct-variable product.
    monomial : list of (int, int)
        Each entry is (var_idx, exponent) for x_i^n, n integer ≥ 2.
    general_nl : list of Expression
        Nonlinear expression nodes that are neither bilinear, trilinear,
        higher-order multilinear, nor monomial (e.g., sin, cos, exp, log,
        sqrt, tan, abs).
    term_incidence : dict[int, set[int]]
        Maps flat variable index → set of term indices (into the combined bilinear +
        trilinear + multilinear list) that the variable appears in. Term indices
        are assigned in product-term discovery order and are used for vertex-cover
        computation.
    partition_candidates : list[int]
        Sorted list of flat variable indices appearing in any bilinear,
        trilinear, or higher-order multilinear product.  These are the
        candidates for domain partitioning in AMP.
        (Monomials are convex/treated separately; general_nl may also be candidates
        but are currently excluded from partitioning as AMP focuses on polynomial terms.)
    """

    bilinear: list[tuple[_VarIdx, _VarIdx]] = field(default_factory=list)
    trilinear: list[tuple[_VarIdx, _VarIdx, _VarIdx]] = field(default_factory=list)
    multilinear: list[tuple[_VarIdx, ...]] = field(default_factory=list)
    monomial: list[tuple[_VarIdx, int]] = field(default_factory=list)
    fractional_power: list[tuple[_VarIdx, float]] = field(default_factory=list)
    # Products of a linear variable with a fractional-power factor, recorded as
    # ``(linear_var_idx, (base_var_idx, exponent))``.  The MILP relaxation lifts
    # the fractional power to an aux column and adds a McCormick envelope on the
    # resulting (linear, aux) bilinear product.
    bilinear_with_fp: list[tuple[_VarIdx, tuple[_VarIdx, float]]] = field(default_factory=list)
    # Ratios of products ``(c·Πx_i)/(Πy_j)`` recorded as
    # ``((num_var_indices), (den_var_indices))`` (sorted, distinct).  The MILP
    # relaxation lifts each via the linear-fractional ``r·q = m`` identity
    # (issue #185); the embedded numerator/denominator products are also recorded
    # as bilinear/trilinear/multilinear terms so they receive McCormick envelopes.
    ratio_of_products: list[tuple[tuple[_VarIdx, ...], tuple[_VarIdx, ...]]] = field(
        default_factory=list
    )
    general_nl: list[Expression] = field(default_factory=list)
    term_incidence: dict[_VarIdx, set[int]] = field(default_factory=dict)
    partition_candidates: list[_VarIdx] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers: flat index extraction
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting flat index of a variable in the stacked x vector.

    Delegates to the model's memoized prefix-sum offset table
    (``Model._flat_var_offset``), turning the classifier's per-term flat-index
    resolution from O(n·terms) into O(n + terms) — the quadratic summation here
    was the dominant uninterruptible root-setup overrun on large factorable
    models (issues #507, #654).
    """
    return model._flat_var_offset(var)


def _get_flat_index(expr: Expression, model: Model) -> int | None:
    """Return the flat variable index for a scalar Variable or IndexExpression.

    Returns None if the expression is not a scalar variable reference.

    #941: this used to open-code the index arithmetic and took negatives
    literally, so ``v[-1]`` resolved to ``base_offset - 1`` — a valid slot
    belonging to a *different* variable. The bilinear catalog then registered a
    pair that does not exist in the model, and the relaxation built from it cut
    off the true optimum under a ``gap_certified=True`` label. It now shares one
    resolver with every other structure layer.
    """
    return resolve_scalar_slot(expr, model)


def _structural_children(e: Expression) -> tuple[Expression, ...] | None:
    """Every operand of ``e``; ``None`` for a node type this module does not know."""
    if isinstance(e, (BinaryOp, MatMulExpression)):
        return (e.left, e.right)
    if isinstance(e, UnaryOp):
        return (e.operand,)
    if isinstance(e, FunctionCall):
        return tuple(e.args)
    if isinstance(e, IndexExpression):
        return (e.base,)
    if isinstance(e, SumExpression):
        return (e.operand,)
    if isinstance(e, SumOverExpression):
        return tuple(e.terms)
    if isinstance(e, (Constant, Parameter, Variable)):
        return ()
    return None


def _is_variable_free(e: Expression, memo: dict[int, bool]) -> bool:
    """True iff ``e`` references no decision variable.

    Memoized on ``id`` (a pure structural predicate) and iterative, so a deep or
    heavily shared DAG costs O(unique nodes) with no recursion-depth hazard. An
    unknown node type counts as variable-bearing: the conservative answer, since
    every caller uses ``False`` only to route a model to the thorough Python walk
    or to expand a node element-wise.
    """
    stack: list[tuple[Expression, bool]] = [(e, False)]
    while stack:
        node, expanded = stack.pop()
        nid = id(node)
        if nid in memo:
            continue
        if isinstance(node, (Constant, Parameter)):
            memo[nid] = True
            continue
        if isinstance(node, Variable):
            memo[nid] = False
            continue
        kids = _structural_children(node)
        if kids is None:
            memo[nid] = False
            continue
        if expanded:
            memo[nid] = all(memo[id(c)] for c in kids)
            continue
        stack.append((node, True))
        stack.extend((c, False) for c in kids)
    return memo[id(e)]


def _is_array_nonlinear(expr: Expression, memo: dict[int, bool]) -> bool:
    """A node that is nonlinear only *element-wise* over an array (#1591).

    ``@`` with a variable on both sides (``x @ Q @ x`` -- scalar-valued but a
    contraction of array operands), or an array-shaped ``*`` / ``**`` over
    variables (``x * (Q @ x)``, ``x ** 3``). Neither classifier recognizes such a
    node as a whole: its scalar terms only exist once it is expanded.
    """
    if isinstance(expr, MatMulExpression):
        return not _is_variable_free(expr.left, memo) and not _is_variable_free(expr.right, memo)
    if isinstance(expr, BinaryOp) and expr.op in ("*", "**"):
        if static_shape(expr) == ():
            return False
        if expr.op == "**":
            return not _is_variable_free(expr.left, memo)
        return not _is_variable_free(expr.left, memo) and not _is_variable_free(expr.right, memo)
    return False


def _rebalance_additive(expr: Expression) -> Expression:
    """Rewrite the additive skeleton of a scalarized element as a balanced ``+`` tree.

    :mod:`_relax.scalarize` spells a contraction two ways -- a left-nested ``+``
    chain (``_elem_matmul``) or a flat :class:`SumOverExpression` (a reduction,
    a scalar ``@``). :func:`distribute_products` descends neither usefully: it
    does not enter a ``SumOverExpression`` at all, so ``x[k] * Σ_i Q[k,i] x[i]``
    stays one undistributable product, and it recurses once per link of a chain,
    so an ``n``-term contraction costs ``n`` stack frames. Flattening every
    ``+``/``-``/``neg``/``SumOver`` run (iteratively) and rebuilding it balanced
    fixes both: the result is algebraically identical, distributes completely, and
    is ``O(log n)`` deep. Used only on freshly scalarized elements (#1591), whose
    nodes no ``id()``-keyed map refers to.
    """

    def leaf(node: Expression) -> Expression:
        if isinstance(node, BinaryOp) and node.op in ("*", "/", "**"):
            left = _rebalance_additive(node.left)
            right = _rebalance_additive(node.right)
            if left is node.left and right is node.right:
                return node
            return BinaryOp(node.op, left, right)
        return node

    def is_link(node: Expression) -> bool:
        return (
            isinstance(node, SumOverExpression)
            or (isinstance(node, BinaryOp) and node.op in ("+", "-"))
            or (isinstance(node, UnaryOp) and node.op == "neg")
        )

    if not is_link(expr):
        return leaf(expr)
    parts: list[Expression] = []
    stack: list[tuple[Expression, bool]] = [(expr, False)]
    while stack:
        node, negated = stack.pop()
        if isinstance(node, SumOverExpression):
            stack.extend((t, negated) for t in reversed(node.terms))
        elif isinstance(node, BinaryOp) and node.op in ("+", "-"):
            stack.append((node.right, negated if node.op == "+" else not negated))
            stack.append((node.left, negated))
        elif isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((node.operand, not negated))
        else:
            term = leaf(node)
            parts.append(UnaryOp("neg", term) if negated else term)
    if not parts:
        return Constant(0.0)
    while len(parts) > 1:
        paired: list[Expression] = [
            BinaryOp("+", parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)
        ]
        if len(parts) % 2:
            paired.append(parts[-1])
        parts = paired
    return parts[0]


def _contains_expandable_square(model: Model) -> bool:
    """Return True if Python classification should distribute a non-leaf product.

    Covers two shapes the Rust arena classifier does NOT distribute, both of
    which hide bilinear/monomial cross-terms behind an additive composite:

    * ``(expr)**2`` with a non-leaf base, e.g. ``(x_i - x_j)**2``; and
    * an explicit self-/cross-product of additive composites written without the
      power operator, e.g. ``(x_i - x_j) * (x_i - x_j)`` (the form MINLPLib's
      circle-packing distance constraints use). The Rust classifier sees the
      ``*`` but never expands the ``-`` inside, so it misses the ``x_i*x_j``
      cross-term; the linearizer then raises "Bilinear (i,j) not in map" and the
      whole constraint is dropped, collapsing the relaxation bound to a trivial
      value (kall_congruentcircles_* never certified for exactly this reason).

    Routing such models to the Python classifier, which distributes via
    ``_distribute_mul``, recovers the full term set. Classification runs once per
    solve (not per node), so the cost of the Python path here is negligible.

    #1591: it also covers **array-valued** nonlinear structure, which the Rust
    arena classifier cannot scalarize and therefore catalogs as *nothing*:

    * a ``@`` whose operands both carry a variable (``x @ Q @ x``, ``x @ y``,
      ``(A @ x) @ (B @ y)``) -- the Rust ``MatMul`` arm only recurses; and
    * an array-shaped ``*`` / ``**`` over variables (``x * (Q @ x)``,
      ``x ** 3``) -- the Rust product/power arms resolve only *scalar*
      variable factors, so an array factor makes the whole product vanish.

    Silent in two ways: alone, such a model reached the Python walk only through
    the "catalog empty but model nonlinear" cross-check; next to any scalar term,
    the Rust route returned that term and nothing else -- an incomplete catalog
    reported as a complete one.
    """

    def _is_additive_composite(e: Expression) -> bool:
        """A ``+``/``-`` node whose distribution can expose product cross-terms."""
        return isinstance(e, BinaryOp) and e.op in ("+", "-")

    # Memoize on ``id(expr)``: this is a *pure structural* predicate — it inspects
    # only op / constant structure, never variable bounds — so an id-keyed cache is
    # bound-neutral and cannot go stale during the walk. Dense-quadratic models share
    # subexpressions heavily (qap: a 225-variable assignment objective is a sum of
    # ~n^2 products over a small variable set), so the naive ``visit(left) or
    # visit(right)`` re-walks shared nodes combinatorially — minutes of wall /
    # RecursionError on qap. With the memo each unique node is visited once -> O(nodes).
    _seen: dict[int, bool] = {}
    _vf_memo: dict[int, bool] = {}

    def _decide_here(expr: Expression) -> bool | None:
        """The node's own verdict, when it can be reached without its children."""
        if _is_array_nonlinear(expr, _vf_memo):
            return True
        if isinstance(expr, BinaryOp):
            if (
                expr.op == "**"
                and isinstance(expr.right, Constant)
                and float(expr.right.value) == 2.0
                and _get_flat_index(expr.left, model) is None
            ):
                return True
            if expr.op == "*" and (
                _is_additive_composite(expr.left) or _is_additive_composite(expr.right)
            ):
                return True
        return None

    def _children(expr: Expression) -> tuple[Expression, ...]:
        if isinstance(expr, BinaryOp):
            return (expr.left, expr.right)
        if isinstance(expr, UnaryOp):
            return (expr.operand,)
        if isinstance(expr, FunctionCall):
            return tuple(expr.args)
        if isinstance(expr, IndexExpression):
            # A plain ``v[i]`` is a variable reference, not a composite.
            return () if isinstance(expr.base, Variable) else (expr.base,)
        if isinstance(expr, SumExpression):
            return (expr.operand,)
        if isinstance(expr, SumOverExpression):
            return tuple(expr.terms)
        if isinstance(expr, MatMulExpression):
            return (expr.left, expr.right)
        return ()

    def visit(root: Expression) -> bool:
        """Iterative post-order walk.

        The memo above bounds repeated WORK but not recursion DEPTH: a lifted
        model's objective is a left-deep ``((a + b) + c) + ...`` chain as long as
        its term count, so a recursive visitor needs one frame per term and dies
        with RecursionError past ~1000 of them. That exception is an ``Exception``
        subclass, so the reformulation-adoption guard in ``solve_model`` caught it
        and silently reported "no reformulation available" -- the pass was skipped
        on exactly the biggest models it was written for, with nothing but a
        ``logger.debug`` to say so. An explicit stack removes the depth limit; the
        memo keeps it O(unique nodes).
        """
        stack: list[tuple[Expression, bool]] = [(root, False)]
        while stack:
            expr, expanded = stack.pop()
            eid = id(expr)
            if eid in _seen:
                continue
            if expanded:
                _seen[eid] = any(_seen.get(id(c), False) for c in _children(expr))
                continue
            here = _decide_here(expr)
            if here is not None:
                _seen[eid] = here
                continue
            kids = _children(expr)
            if not kids:
                _seen[eid] = False
                continue
            stack.append((expr, True))
            stack.extend((c, False) for c in kids)
        return _seen[id(root)]

    if model._objective is not None and visit(model._objective.expression):
        return True
    return any(visit(constraint.body) for constraint in model._constraints)


# ---------------------------------------------------------------------------
# Helpers: product-tree decomposition
# ---------------------------------------------------------------------------


def _collect_product_factors(expr: Expression, model: Model) -> list[int] | None:
    """Try to decompose a pure product tree into a list of flat variable indices.

    Handles: Variable * Variable, (Var * Var) * Var, Var[i] * Var[j], etc.
    Returns None if the expression contains non-variable leaves (e.g., constants,
    general functions).  Constant scale factors are NOT handled here — they belong
    in the coefficient extraction, not term classification.
    """
    indices: list[int] = []

    def _visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return _visit(e.left) and _visit(e.right)
        # Unary negation is just a sign on a scalar factor, never a variable
        # factor — make it transparent.  A leading negative coefficient such as
        # ``-c*x*y`` parses as ``((neg(Constant(c)) * x) * y)``; without this the
        # whole product term is rejected and silently dropped from classification.
        if isinstance(e, UnaryOp) and e.op == "neg":
            return _visit(e.operand)
        flat = _get_flat_index(e, model)
        if flat is not None:
            indices.append(flat)
            return True
        # Constant multiplier: skip it (treat as scaling, not a new variable)
        if isinstance(e, Constant):
            return True
        return False

    if _visit(expr):
        # Filter out duplicates introduced by constants (empty index list)
        var_indices = indices  # may have duplicates if e.g. x*x
        if len(var_indices) >= 2:
            return var_indices
    return None


def _distribute_mul(left: Expression, right: Expression) -> Expression:
    """Distribute ``left * right`` where BOTH are already fully distributed.

    Recurses only over the additive (``+``/``-``) structure of the operands —
    never over their already-flat product leaves — so it builds exactly the
    output sum-of-products with no re-walking. The result tree's additive terms
    and their signs match the naive ``a*c + b*c`` expansion.
    """
    if isinstance(right, BinaryOp) and right.op in ("+", "-"):
        return BinaryOp(
            right.op,
            _distribute_mul(left, right.left),
            _distribute_mul(left, right.right),
        )
    if isinstance(left, BinaryOp) and left.op in ("+", "-"):
        return BinaryOp(
            left.op,
            _distribute_mul(left.left, right),
            _distribute_mul(left.right, right),
        )
    return BinaryOp("*", left, right)


def _distribute_power_over_product(base: Expression, n: int) -> Expression | None:
    """Expand ``(a * b)**n`` → ``a**n * b**n`` for integer ``n >= 2`` over a
    purely multiplicative ``base``.

    This is an *exact* algebraic identity only because the factors are
    multiplied — it must never be applied across a sum, so a sum (or any
    non-multiplicative node: division, transcendental call, negation) makes the
    function return ``None`` and the caller leaves the original power node
    intact.  Folding a power of a product into a product of powers lets the
    downstream factor collector (``_collect_extended_factors``) expand each
    ``x**k`` into its repeated flat-variable factors instead of stranding the
    whole power-of-product in the un-linearizable ``extra`` bucket — which is
    what silently dropped nvs06's defining constraint ``w * (x0*x1)**4 == …``.

    Constants fold to ``value**n``; a flat-indexable leaf becomes ``leaf**n``; a
    nested integer power ``(b**m)**n`` collapses to ``b**(m*n)`` (recursing so a
    nested product base also expands).
    """
    if isinstance(base, Constant):
        return Constant(float(base.value) ** n)
    if isinstance(base, (Variable, IndexExpression)):
        return BinaryOp("**", base, Constant(float(n)))
    if isinstance(base, BinaryOp):
        if base.op == "*":
            left = _distribute_power_over_product(base.left, n)
            right = _distribute_power_over_product(base.right, n)
            if left is None or right is None:
                return None
            return BinaryOp("*", left, right)
        if base.op == "**" and isinstance(base.right, Constant):
            m = float(base.right.value)
            m_int = int(m)
            if m == m_int and m_int >= 1:
                # (b**m)**n → b**(m*n); recurse so a product base under the
                # inner power still expands to a product of powers.
                return _distribute_power_over_product(base.left, m_int * n)
    return None


# Saturate the term estimate so a deeply nested blowup cannot overflow before it
# trips a limit.
_TERM_CAP = 1 << 40


def estimate_distributed_terms(expr: Expression) -> int:
    """Estimate how many additive terms :func:`distribute_products` would yield.

    A product multiplies its operands' term counts, an integer power raises the
    base's, a sum adds them; every opaque leaf (variable, call, division,
    constant) counts as one.  Saturated at :data:`_TERM_CAP`.

    Monotone up the tree — every node's estimate is >= each child's — which is
    what lets :func:`distribute_products` decide the whole expression is
    affordable with a single check at the root.

    Memoized on node identity (#1456): the estimate is a pure function of the
    subtree, and an expression DAG with shared subtrees (``from_nl`` common
    subexpressions, or a :func:`distribute_products` result, whose terms share
    their factor subtrees) made the unmemoized recursion cost the *tree* size,
    not the DAG size. Same arithmetic, same saturation, same value.
    """
    return _estimate_terms(expr, {}, False)


def _estimate_terms(expr: Expression, memo: dict[int, int], unfolded: bool) -> int:
    """``unfolded=True`` is :func:`_unfolded_term_bound`'s variant: a quotient by
    a nonzero scalar constant counts its numerator's terms (the fold rewrites
    ``(a + b) / 2`` as ``0.5*a + 0.5*b``); every other node as above."""
    hit = memo.get(id(expr))
    if hit is not None:
        return hit
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            est = min(
                _estimate_terms(expr.left, memo, unfolded)
                + _estimate_terms(expr.right, memo, unfolded),
                _TERM_CAP,
            )
        elif expr.op == "*":
            est = min(
                _estimate_terms(expr.left, memo, unfolded)
                * _estimate_terms(expr.right, memo, unfolded),
                _TERM_CAP,
            )
        elif expr.op == "**" and isinstance(expr.right, Constant):
            n = float(expr.right.value)
            n_int = int(n)
            if n == n_int and n_int >= 1:
                est = min(int(_estimate_terms(expr.left, memo, unfolded) ** n_int), _TERM_CAP)
            else:
                est = 1
        elif unfolded and expr.op == "/" and _scalar_constant(expr.right) not in (None, 0.0):
            est = _estimate_terms(expr.left, memo, unfolded)
        else:
            est = 1
    elif isinstance(expr, UnaryOp):
        est = _estimate_terms(expr.operand, memo, unfolded)
    else:
        est = 1
    memo[id(expr)] = est
    return est


def _unfolded_term_bound(expr: Expression) -> int:
    """An upper bound on ``estimate_distributed_terms(fold_affine_constants(expr))``
    that skips the fold (~80% of the cost of computing that value exactly:
    measured 9.1 s of 11.0 s on telecomsp_nor_sun's 21,271 bodies).

    Why it bounds: the fold only rewrites maximal affine forms (``+``, ``-``,
    ``neg``, product or quotient by a scalar constant), replacing one by a sum
    over its *distinct* atoms with merged coefficients plus at most one constant.
    The estimate of an affine form is the sum of its leaves' estimates (a scalar
    constant factor counts 1, and here so does a constant divisor); merging atoms,
    dropping zero coefficients and collapsing constants only remove leaves, and
    each atom is itself folded, so by induction its estimate does not grow.
    ``**`` and ``*`` are monotone in their operands; ``_TERM_CAP`` saturation is
    monotone.
    """
    return _estimate_terms(expr, {}, True)


# Ceiling on the number of additive terms a single ``distribute_products`` call
# will expand to.  Symbolic distribution is exponential in the nesting depth of
# sums-inside-products, and NOTHING else bounds it: the pre-solve structural
# scans (``has_factorable_work`` and the integer-product / quadratic detectors)
# distribute the raw model body purely to look for a pattern, with no size check
# and no deadline -- these passes run inside ``solve_model`` before branch and
# bound starts and neither accept nor check one, so the time limit cannot reach
# them.  ``johnall`` (MINLPLib) asks this path for an expansion of 3.19e9 terms
# and overran a 20 s ``time_limit`` by 44 minutes.
#
# Set from measurement, not taste.  Surveyed over 1610 MINLPLib instances
# (5,072,187 distribute calls), the largest expansion any instance legitimately
# asks for is 998,002 terms (``truck``); the cheapest pathology is ``saa_2`` at
# 2.18e9, ~2200x higher.  This budget sits in that gap: it truncates 2 of 1610
# instances -- ``johnall`` and ``saa_2``, both >2000x over -- and leaves every
# other corpus instance expanded exactly as before.  So it is a backstop against
# a pathology, not a tuning knob trading capability for speed.
#
# It bounds THIS mechanism, not the pre-solve scan as a whole.  Distribution
# costs ~3.6 us/term at this size and grows superlinearly (measured 0.43 us/term
# at 1.7e3 terms, 5.21 us/term at 5.8e6), so a single at-budget call still costs
# ~4 s.  Two other scan paths blow the same time limit by different mechanisms
# and are untouched by this constant: a whole-model-sized ``eigvalsh`` per
# ``sqrt`` node in ``convexity.patterns.is_homogeneous_psd_quadratic``
# (``glider400``), and ``binary_multilinear_reform._poly_add`` (``hadamard_9``,
# which still overran 300 s against a 60 s limit with this budget in force).
# Both were fixed the same way in #1456 -- a deterministic bound on the work,
# not a wall-clock deadline, so that what a pre-solve pass recognizes does not
# depend on machine load (which would make the bound-neutral verification
# regime in CLAUDE.md sec.5 unenforceable on exactly these instances).
#
# Over budget the expression is returned with its affordable subtrees distributed
# and the offending product left intact -- ALGEBRAICALLY IDENTICAL either way, so
# no constraint or objective changes meaning.  What is lost is pattern
# recognition: a detector sees less structure, a relaxation drops a term it
# cannot linearize (a weaker bound, never a wrong one), a reformulation pass
# declines to fire.  Every consumer was audited to abstain rather than conclude
# -- with ONE exception, ``factorable_reform._has_unbounded_nonlinear_term``,
# whose ``False`` *enables* a rewrite and which therefore consults
# :func:`distribution_exceeds_budget` and fails closed.
_DISTRIBUTE_TERM_BUDGET = 1 << 20  # 1,048,576 terms; 1.05x the corpus maximum


def distribution_exceeds_budget(expr: Expression) -> bool:
    """True if :func:`distribute_products` would refuse to fully expand *expr*.

    For callers whose "found nothing" answer is load-bearing: a partial
    distribution means "could not look", not "looked and there was nothing", and
    such a caller must fail closed rather than read the two as the same.
    """
    return estimate_distributed_terms(expr) > _DISTRIBUTE_TERM_BUDGET


# Ceiling on the terms ONE PASS may spend distributing EVERY body of a model
# (#1456, the johnall regression of 2026-10-09).
#
# ``_DISTRIBUTE_TERM_BUDGET`` bounds one call, and a pass that distributes every
# constraint body makes one call per body -- so the per-call budget bounds an
# iteration, never the pass. ``johnall`` is exactly that shape: 190 constraint
# bodies of 129 DAG nodes, each asking for 16.8 M terms, each therefore spending
# the full per-call budget (611,757 output DAG nodes, ~1.8 s apiece) -- ~340 s of
# distribution per pass before a single detector walks the result, and the
# integer-product detector then walked each result as a *tree* (8.26 M nodes, the
# output shares its factor subtrees). ``has_nonconvex_integer_bilinear`` never
# returned against ``time_limit=10``/``20``.
#
# Set from measurement (``sum_body min(est, _DISTRIBUTE_TERM_BUDGET)`` over all
# 1610 MINLPLib instances, 2026-10-09): the largest legitimate total is 4,107,430
# terms (``acopf_caseactivsg70k_qcqp``, 1,016,477 bodies -- a total that grows
# with model size, which is fine); the next two are ``johnall`` at 200,278,212
# and ``saa_2`` at 2,097,170,814. 2**24 = 16,777,216 sits in that gap, 4.1x above
# the corpus maximum and 11.9x below the cheapest pathology, so it truncates
# exactly those two instances -- the same two the per-call budget truncates --
# and changes nothing anywhere else. A backstop, not a tuning knob.
#
# A deterministic work count, not a clock: what a pass recognises must not
# depend on machine load (#912; CLAUDE.md §5's bound-neutral regime).
_MODEL_DISTRIBUTE_TERM_BUDGET = 1 << 24


def _model_bodies(model: Model):
    if model._objective is not None:
        yield model._objective.expression
    for c in model._constraints:
        if isinstance(c, Constraint):
            yield c.body


def model_distribution_terms(model: Model) -> int:
    """Terms a pass spends distributing every body of *model* once.

    Each body is charged what :func:`distribute_products` would actually spend on
    it -- its estimate after the same affine fold, capped at the per-call
    :data:`_DISTRIBUTE_TERM_BUDGET` (beyond which the call stops distributing).
    Linear in the model's DAG size; no body is distributed to compute it.
    """
    return _model_distribution_terms(model, exact=True)


def _model_distribution_terms(model: Model, *, exact: bool) -> int:
    if not exact:
        return sum(
            min(_unfolded_term_bound(body), _DISTRIBUTE_TERM_BUDGET)
            for body in _model_bodies(model)
        )
    total = 0
    for body in _model_bodies(model):
        est = estimate_distributed_terms(fold_affine_constants(body))
        total += min(est, _DISTRIBUTE_TERM_BUDGET)
    return total


def model_distribution_exceeds_budget(model: Model, *, pass_name: str) -> bool:
    """True when a pass that distributes every body of *model* must abstain.

    The deterministic work budget for the whole-model distributing pre-solve
    passes (:data:`_MODEL_DISTRIBUTE_TERM_BUDGET`). A pass that gets ``True``
    takes its existing "found nothing / model unchanged" path. Abstention is
    logged (#1456 item 4): it means weaker structure recognition, and a reader
    comparing two runs must be able to see that.
    """
    # The fold-free upper bound settles every model that fits (all MINLPLib
    # instances but johnall and saa_2) at a fifth of the cost; only a model it
    # cannot clear pays for the exact, folded count.
    if _model_distribution_terms(model, exact=False) <= _MODEL_DISTRIBUTE_TERM_BUDGET:
        return False
    total = model_distribution_terms(model)
    if total <= _MODEL_DISTRIBUTE_TERM_BUDGET:
        return False
    logger.warning(
        "%s: distributing every body would cost %s terms, over the %s-term "
        "per-pass budget; the pass abstains and leaves the model unchanged "
        "(structure may go unrecognized and bounds may be weaker; #1456)",
        pass_name,
        f"{total:,}",
        f"{_MODEL_DISTRIBUTE_TERM_BUDGET:,}",
    )
    return True


def distribute_bodies(
    bodies,
    *,
    pass_name: str,
    protected_squares: frozenset[int] | None = None,
):
    """Yield :func:`distribute_products` of each of *bodies*, under the per-pass
    budget :data:`_MODEL_DISTRIBUTE_TERM_BUDGET` (#1456).

    For a pass that distributes every body of a model. Each body is folded and
    estimated first (linear in its DAG size, no distribution); when the total the
    pass would spend -- each body charged ``min(estimate, per-call budget)``, as
    in :func:`model_distribution_terms` -- fits the per-pass budget, every body
    is distributed exactly as :func:`distribute_products` distributes it. Over
    it, a body that fits the per-call budget is still distributed in full, and a
    body that does not is yielded folded but undistributed (logged once, here)
    rather than partially distributed to the per-call budget.

    That second output is the shape a partially distributed body already has --
    an algebraically identical expression with products left intact -- so every
    consumer audited for the per-call budget (see its definition) handles it the
    same way: as structure not recognised, never as a wrong conclusion. The
    decision is a function of the model alone: a deterministic work count, not
    a clock.
    """
    prepared = []
    total = 0
    for body in bodies:
        folded = fold_affine_constants(body, protected_squares)
        est = estimate_distributed_terms(folded)
        prepared.append((folded, est))
        total += min(est, _DISTRIBUTE_TERM_BUDGET)
    whole = total <= _MODEL_DISTRIBUTE_TERM_BUDGET
    if not whole:
        logger.warning(
            "%s: distributing every body would cost %s terms, over the %s-term "
            "per-pass budget; the %d bodies the per-call budget cannot afford are "
            "left undistributed (algebraically identical, but structure may go "
            "unrecognized and bounds may be weaker; #1456)",
            pass_name,
            f"{total:,}",
            f"{_MODEL_DISTRIBUTE_TERM_BUDGET:,}",
            sum(1 for _f, e in prepared if e > _DISTRIBUTE_TERM_BUDGET),
        )
    for folded, est in prepared:
        yield _distribute_folded(folded, est, protected_squares, whole)


#: :class:`~discopt._work_budget.WorkBudget` kind for a lazy whole-model pass:
#: distributed terms, each body charged ``min(estimate, per-call budget)``.
DISTRIBUTED_TERMS = "distributed_terms"


def pass_distribution_budget() -> WorkBudget:
    """A fresh per-pass budget of :data:`_MODEL_DISTRIBUTE_TERM_BUDGET`
    distributed terms, for :func:`distribute_charged`."""
    return WorkBudget({DISTRIBUTED_TERMS: _MODEL_DISTRIBUTE_TERM_BUDGET})


def distribute_charged(expr: Expression, budget: WorkBudget) -> Expression | None:
    """:func:`distribute_products` of *expr*, charged to a per-pass *budget*.

    The lazy form of :func:`distribute_bodies`, for a pass that may stop early
    (a scan that returns on its first hit) and so must not fold and estimate
    every body up front. The body is charged what :func:`distribute_products`
    spends on it -- its folded estimate capped at the per-call
    :data:`_DISTRIBUTE_TERM_BUDGET`, the same charge as
    :func:`model_distribution_terms` -- and the result is identical to
    :func:`distribute_products`. Returns ``None``, charging nothing, when the
    body does not fit what is left: the caller abstains. A deterministic work
    count over the bodies in model order, not a clock (#1456).
    """
    folded = fold_affine_constants(expr)
    est = estimate_distributed_terms(folded)
    cost = min(est, _DISTRIBUTE_TERM_BUDGET)
    left = budget.remaining(DISTRIBUTED_TERMS)
    if left is not None and cost > left:
        return None
    budget.charge(DISTRIBUTED_TERMS, cost)
    return _distribute_folded(folded, est, None, True)


def distribute_products(
    expr: Expression, protected_squares: frozenset[int] | None = None
) -> Expression:
    """Recursively distribute multiplication over addition/subtraction, up to a
    term budget (:data:`_DISTRIBUTE_TERM_BUDGET`).

    Beyond the budget the offending product is left undistributed rather than
    expanded; the result is algebraically identical, so this is a loss of
    recognizable structure, never of correctness.  See the budget's definition
    for why it exists and what depends on it.
    """
    expr = fold_affine_constants(expr, protected_squares)
    return _distribute_folded(expr, estimate_distributed_terms(expr), protected_squares, True)


def _distribute_folded(
    expr: Expression,
    est: int,
    protected_squares: frozenset[int] | None,
    partial: bool,
) -> Expression:
    """Distribute an already-folded *expr* whose term estimate is *est*.

    ``partial=False`` is the per-pass budget's mode (:func:`distribute_bodies`):
    a body the per-call budget cannot afford is returned folded but otherwise
    undistributed, instead of being distributed up to the per-call budget --
    which is what costs a whole-model pass ``n_bodies`` times that budget.
    """
    if est <= _DISTRIBUTE_TERM_BUDGET:
        return _distribute_unbudgeted(expr, protected_squares)
    if not partial:
        return expr
    result = _distribute_within_budget(expr, protected_squares, {}, [_DISTRIBUTE_TERM_BUDGET])
    logger.warning(
        "distribute_products: %s estimated terms exceeds the %s-term budget; the "
        "oversized products were left undistributed (algebraically identical, but "
        "structure may go unrecognized and bounds may be weaker)",
        f"{est:,}",
        f"{_DISTRIBUTE_TERM_BUDGET:,}",
    )
    return result


def _scalar_constant(expr: Expression) -> float | None:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1 or not math.isfinite(float(values[0])):
        return None
    return float(values[0])


def _affine_atom_key(atom: Expression) -> tuple:
    """Identity key under which two atoms of an affine form may be combined.

    A scalar variable is keyed by the variable object, an integer-indexed element
    of a variable by ``(variable, index)``; every other atom is keyed by its own
    node identity, so structurally equal but distinct subtrees are never merged
    (conservative: a missed merge only leaves the expression as it was).
    """
    if isinstance(atom, Variable):
        return ("v", id(atom))
    if isinstance(atom, IndexExpression) and isinstance(atom.base, Variable):
        idx = atom.index
        if type(idx) is int:
            return ("i", id(atom.base), idx)
        if isinstance(idx, tuple) and all(type(i) is int for i in idx):
            return ("i", id(atom.base), idx)
    return ("n", id(atom))


def _is_affine_link(expr: Expression, protected: frozenset[int] | None) -> bool:
    """``+``/``-``/``neg`` or a product/quotient by a scalar constant literal."""
    if protected is not None and id(expr) in protected:
        return False
    if isinstance(expr, UnaryOp):
        return expr.op == "neg"
    if not isinstance(expr, BinaryOp):
        return False
    if expr.op in ("+", "-"):
        return True
    if expr.op == "*":
        return _scalar_constant(expr.left) is not None or _scalar_constant(expr.right) is not None
    if expr.op == "/":
        rv = _scalar_constant(expr.right)
        return rv is not None and rv != 0.0
    return False


def _affine_walk(expr: Expression, protected: frozenset[int] | None, visit) -> None:
    """Call ``visit(kind, node, scale)`` on every leaf of the maximal affine form.

    ``kind`` is ``"const"`` (with the leaf's value folded into ``scale``) or
    ``"atom"``. ``scale`` is an exact ``Fraction``.
    """
    stack: list[tuple[Expression, Fraction]] = [(expr, Fraction(1))]
    while stack:
        node, scale = stack.pop()
        const = _scalar_constant(node)
        if const is not None:
            visit("const", node, scale * Fraction(const))
            continue
        if not _is_affine_link(node, protected):
            visit("atom", node, scale)
            continue
        if isinstance(node, UnaryOp):
            stack.append((node.operand, -scale))
            continue
        assert isinstance(node, BinaryOp)  # _is_affine_link admits only these two
        if node.op in ("+", "-"):
            stack.append((node.right, scale if node.op == "+" else -scale))
            stack.append((node.left, scale))
            continue
        lc = _scalar_constant(node.left)
        rc = _scalar_constant(node.right)
        if node.op == "*" and lc is not None:
            stack.append((node.right, scale * Fraction(lc)))
        elif node.op == "*" and rc is not None:
            stack.append((node.left, scale * Fraction(rc)))
        elif node.op == "/" and rc is not None and rc != 0.0:
            stack.append((node.left, scale / Fraction(rc)))
        else:  # unreachable: _is_affine_link admitted the node
            raise AssertionError(f"not an affine link: {node!r}")


def _fold_affine_root(expr: Expression, protected: frozenset[int] | None) -> Expression:
    """Combine like atoms and constants of the affine form rooted at ``expr``.

    Rebuilt only when something actually combines (two or more constant leaves,
    or an atom repeated); otherwise ``expr`` is returned with identity intact.
    """
    n_const = 0
    seen: set[tuple] = set()
    repeated = False

    def count(kind: str, node: Expression, scale: Fraction) -> None:
        nonlocal n_const, repeated
        if kind == "const":
            n_const += 1
            return
        key = _affine_atom_key(node)
        if key in seen:
            repeated = True
        seen.add(key)

    _affine_walk(expr, protected, count)
    if n_const < 2 and not repeated:
        return expr

    const_total = Fraction(0)
    coeffs: dict[tuple, list] = {}

    def collect(kind: str, node: Expression, scale: Fraction) -> None:
        nonlocal const_total
        if kind == "const":
            const_total += scale
            return
        key = _affine_atom_key(node)
        slot = coeffs.get(key)
        if slot is None:
            coeffs[key] = [node, scale]
        else:
            slot[1] += scale

    _affine_walk(expr, protected, collect)
    out: Expression | None = None
    for atom, coeff in coeffs.values():
        if coeff == 0:
            continue
        mag = abs(coeff)
        term = atom if mag == 1 else BinaryOp("*", Constant(float(mag)), atom)
        if out is None:
            out = term if coeff > 0 else UnaryOp("neg", term)
        else:
            out = BinaryOp("+" if coeff > 0 else "-", out, term)
    if const_total != 0:
        cval = float(const_total)
        if out is None:
            out = Constant(cval)
        else:
            out = BinaryOp("+" if cval > 0 else "-", out, Constant(abs(cval)))
    return out if out is not None else Constant(0.0)


def fold_affine_constants(
    expr: Expression, protected_squares: frozenset[int] | None = None
) -> Expression:
    """Fold each maximal affine subtree's like terms and constants exactly (#1542).

    ``(z + c) - c`` denotes ``z``, but distributing it as written multiplies the
    cancelling constants into every product it meets: ``((z + c) - c) * w``
    becomes ``z*w + c*w - c*w`` and a square of it carries ``c**2`` terms. At
    ``|c| ~ 1e6`` those terms are ``1e12``-scale, and once a downstream consumer
    sums them in floating point (the factorable reform's ``_fr_aux_*`` defining
    rows, the LP coefficients) the cancellation leaves an error of ``ulp(1e12)``
    or more on an ``O(1)`` quantity -- enough to make st_e36's root McCormick LP
    infeasible on a feasible model, certified as infeasible.

    Each maximal ``+``/``-``/``neg``/scalar-constant-scaled subtree is collected
    into ``sum(coeff * atom) + const`` with the coefficients and the constant
    combined in exact rational arithmetic, each rounded once at the end. The
    walk mirrors distribution's: it descends through ``BinaryOp``/``UnaryOp``
    only, never into a protected node or a ``FunctionCall`` (whose ``id()`` keys
    lift maps), and a subtree in which nothing combines keeps its identity.
    """
    memo: dict[tuple[int, bool], Expression] = {}

    def fold(node: Expression, inside_affine: bool) -> Expression:
        key = (id(node), inside_affine)
        hit = memo.get(key)
        if hit is not None:
            return hit
        if protected_squares is not None and id(node) in protected_squares:
            result = node
        elif isinstance(node, BinaryOp):
            link = _is_affine_link(node, protected_squares)
            left = fold(node.left, link)
            right = fold(node.right, link)
            result = (
                node
                if left is node.left and right is node.right
                else BinaryOp(node.op, left, right)
            )
            if link and not inside_affine:
                result = _fold_affine_root(result, protected_squares)
        elif isinstance(node, UnaryOp):
            link = _is_affine_link(node, protected_squares)
            operand = fold(node.operand, link)
            result = node if operand is node.operand else UnaryOp(node.op, operand)
            if link and not inside_affine:
                result = _fold_affine_root(result, protected_squares)
        else:
            result = node
        memo[key] = result
        return result

    return fold(expr, False)


def _distribute_within_budget(
    expr: Expression,
    protected_squares: frozenset[int] | None,
    memo: dict[int, int],
    remaining: list[int],
) -> Expression:
    """Distribute what the budget still affords; leave the rest intact.

    *remaining* is a one-element mutable cell holding the terms left to spend.
    It is a RUNNING TOTAL, not a per-node limit: bounding each node separately
    bounds nothing, because a sum of 3000 sub-products each just under the limit
    still costs 3e9 terms — the exact shape that motivated this budget.  Spending
    from a shared pool caps the whole call.

    Descending rather than refusing the whole expression outright matters: one
    blown-up term in a long sum must not cost the other terms their distribution.
    The traversal is left-to-right, so which terms get distributed depends on
    where the budget runs out — deterministic for a given expression, but not a
    property any caller should lean on.

    Only reached when the root is over budget, so the repeated estimates are off
    the hot path; *memo* keeps them linear anyway.
    """
    # A protected node (issue #155 affine square, issue #358 convex-subexpression
    # lift) is returned with its identity intact, exactly as the unbudgeted walk
    # does. Without this an over-budget protected node would be descended into
    # and rebuilt, and the linearizer's id()-keyed ``composite_var_map`` would no
    # longer resolve it -- silently dropping the lift on precisely the large
    # models this path exists for.
    if protected_squares is not None and id(expr) in protected_squares:
        return expr
    est = memo.get(id(expr))
    if est is None:
        est = memo[id(expr)] = estimate_distributed_terms(expr)
    if est <= remaining[0]:
        remaining[0] -= est
        return _distribute_unbudgeted(expr, protected_squares)
    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        left = _distribute_within_budget(expr.left, protected_squares, memo, remaining)
        right = _distribute_within_budget(expr.right, protected_squares, memo, remaining)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _distribute_within_budget(expr.operand, protected_squares, memo, remaining)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    # A ``*`` or ``**`` the budget cannot afford: distribute inside its operands
    # where affordable, but do not multiply them out.
    if isinstance(expr, BinaryOp):
        left = _distribute_within_budget(expr.left, protected_squares, memo, remaining)
        right = _distribute_within_budget(expr.right, protected_squares, memo, remaining)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    return expr


def _distribute_unbudgeted(
    expr: Expression, protected_squares: frozenset[int] | None = None
) -> Expression:
    """Recursively distribute multiplication over addition/subtraction.

    ``(a + b) * c`` → ``a*c + b*c``;  ``c * (a - b)`` → ``c*a - c*b``.
    ``(a + b)^2`` → ``(a + b) * (a + b)`` before distribution.
    Applied bottom-up so nested distributions resolve.  Other expression
    types are returned with operator-tree shape preserved structurally.

    Operands are distributed exactly once (bottom-up); the multiplication itself
    is then expanded by :func:`_distribute_mul`, which walks only the additive
    structure of the already-distributed operands. The earlier formulation
    re-invoked ``distribute_products`` on every product it constructed, re-walking
    and rebuilding already-flat subtrees — quadratic-to-exponential node creation
    even when the final expansion is small (e.g. a chain of small squared sums
    blew up to tens of millions of throwaway nodes).

    ``protected_squares`` holds ``id()`` values of nodes that must be left intact
    rather than distributed.  Originally these were ``E**2`` affine-square nodes
    (issue #155); it now also covers convex polynomial subexpressions the MILP
    relaxation lifts whole to a single gradient-cut column (issue #358).  In both
    cases distributing the node would re-expand it (catastrophic high-degree
    monomials for a square; loss of the convex-lift identity for a sum) — and
    preserving the node's identity lets the linearizer resolve it through its
    id-keyed ``composite_var_map``.  A protected node of ANY operator is returned
    intact, so the check is at the top rather than only on the ``**`` branch.
    """
    # Any protected node (square lift #155, convex-subexpression lift #358) is
    # returned with its identity intact so its id()-keyed claim survives.
    if protected_squares is not None and id(expr) in protected_squares:
        return expr
    if isinstance(expr, BinaryOp):
        if expr.op == "**" and isinstance(expr.right, Constant):
            # A protected power node (an issue-#155 affine square, or an affine
            # ``(c*x)**n`` lifted to its own scaled-residual envelope column) is
            # returned intact so its ``id()`` still resolves through the
            # linearizer's composite-aux map.
            if protected_squares is not None and id(expr) in protected_squares:
                return expr
            exp_val = float(expr.right.value)
            n_int = int(exp_val)
            if exp_val == n_int and n_int >= 2:
                # (a*b)**n → a**n * b**n when the base is a pure product; leaves a
                # sum base (e.g. (a+b)**n) untouched (helper returns None) so the
                # ``**2`` square-of-sum path and higher sum-powers are unaffected.
                base = _distribute_unbudgeted(expr.left, protected_squares)
                expanded = _distribute_power_over_product(base, n_int)
                if expanded is not None:
                    return expanded
            if exp_val == 2.0:
                left = _distribute_unbudgeted(expr.left, protected_squares)
                return _distribute_mul(left, left)
        left = _distribute_unbudgeted(expr.left, protected_squares)
        right = _distribute_unbudgeted(expr.right, protected_squares)
        if expr.op == "*":
            return _distribute_mul(left, right)
        # Preserve node identity when nothing distributed, so id()-keyed maps
        # (e.g. composite/univariate aux columns) still match the rebuilt tree.
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _distribute_unbudgeted(expr.operand, protected_squares)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    return expr


def _collect_extended_factors(
    expr: Expression, model: Model
) -> tuple[list[int], list[tuple[int, float]]] | None:
    """Decompose a product tree into (flat-variable factors, fractional-power factors).

    Returns ``None`` if the product tree contains non-variable, non-fractional-power
    leaves (e.g., transcendental calls, sums, divisions).  Constant scale factors are
    skipped (handled separately by the linearizer).

    ``var^p`` with non-integer ``p`` and a flat-indexable base is captured as a
    virtual ``(flat_idx, exp)`` factor; integer powers ``var^n`` (n ≥ 2) are
    expanded into ``n`` repeated flat-variable factors so existing bilinear /
    trilinear / monomial handling continues to apply.
    """
    flat_factors: list[int] = []
    fp_factors: list[tuple[int, float]] = []

    def _visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return _visit(e.left) and _visit(e.right)
        if isinstance(e, Constant):
            return True
        # Unary negation is just a sign on a scalar factor (see
        # ``_collect_product_factors``); make it transparent so a leading
        # negative coefficient does not drop the whole product term.
        if isinstance(e, UnaryOp) and e.op == "neg":
            return _visit(e.operand)
        flat = _get_flat_index(e, model)
        if flat is not None:
            flat_factors.append(flat)
            return True
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                exp_val = float(e.right.value)
                n_int = int(exp_val)
                if exp_val == n_int and n_int >= 2:
                    flat_factors.extend([base_flat] * n_int)
                    return True
                if exp_val == n_int and n_int == 1:
                    flat_factors.append(base_flat)
                    return True
                if exp_val != n_int:
                    fp_factors.append((base_flat, exp_val))
                    return True
        return False

    if _visit(expr):
        if len(flat_factors) + len(fp_factors) >= 2:
            return flat_factors, fp_factors
    return None


def extract_single_var_power(expr: Expression, model: Model) -> tuple[int, float] | None:
    """Recognize ``expr`` as ``x**p`` for a single flat-indexable variable ``x``.

    Folds a product / power / ``sqrt`` tree over ONE variable into a single
    ``(flat_idx, exponent)``: e.g. ``sqrt(x)`` → ``(idx, 0.5)``, ``x**3`` →
    ``(idx, 3.0)``, ``x**3 * sqrt(x)`` → ``(idx, 3.5)``. Returns ``None`` for
    anything else (multiple variables, constant scale factors, transcendental
    calls, sums). Constant scale factors are intentionally rejected so callers
    do not silently drop a coefficient.

    This is what lets a reciprocal of a monomial product — ``1/(x**3 * sqrt(x))``
    in nvs08 — be canonicalized to the fractional power ``x**-3.5`` and relaxed,
    rather than dropped as a non-constant division.
    """

    def _visit(e: Expression) -> tuple[int, float] | None:
        flat = _get_flat_index(e, model)
        if flat is not None:
            return (flat, 1.0)
        if isinstance(e, FunctionCall) and e.func_name == "sqrt" and len(e.args) == 1:
            inner = _visit(e.args[0])
            if inner is not None:
                return (inner[0], 0.5 * inner[1])
            return None
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            inner = _visit(e.left)
            if inner is not None:
                return (inner[0], inner[1] * float(e.right.value))
            return None
        if isinstance(e, BinaryOp) and e.op == "*":
            left = _visit(e.left)
            right = _visit(e.right)
            if left is not None and right is not None and left[0] == right[0]:
                return (left[0], left[1] + right[1])
            return None
        return None

    return _visit(expr)


def _ratio_fold_const(expr: Expression) -> float | None:
    """Fold a variable-free subexpression to a scalar, else ``None``.

    Handles literal constants and composite constants (``neg(1e6)``, ``-3*-3``,
    arithmetic over constants) so a numerator scale factor such as gear4's
    ``-1000000`` does not abort product recognition. Conservative: returns
    ``None`` the moment a variable / unhandled node is seen.
    """
    if isinstance(expr, Constant):
        try:
            return float(expr.value)
        except (TypeError, ValueError):
            return None
    if isinstance(expr, UnaryOp):
        sub = _ratio_fold_const(expr.operand)
        if sub is None:
            return None
        if expr.op == "neg":
            return -sub
        if expr.op == "abs":
            return abs(sub)
        return None
    if isinstance(expr, BinaryOp):
        left = _ratio_fold_const(expr.left)
        if left is None:
            return None
        right = _ratio_fold_const(expr.right)
        if right is None:
            return None
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return None if right == 0.0 else left / right
        if expr.op == "**":
            try:
                result = left**right
            except (ValueError, OverflowError, ZeroDivisionError):
                return None
            return None if isinstance(result, complex) else float(result)
    return None


def _collect_ratio_product_vars(expr: Expression, model: Model) -> list[int] | None:
    """Flat variable indices of a pure product ``c·Πx_i`` (constants folded away).

    Returns the list of flat indices (with repeats for integer powers ``x**n``,
    ``2 ≤ n ≤ 4``) or ``None`` if any leaf is not a constant, a flat-indexable
    variable, an integer power thereof, or a division by a constant.
    """
    indices: list[int] = []

    def visit(e: Expression) -> bool:
        if _ratio_fold_const(e) is not None:
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            indices.append(flat)
            return True
        if isinstance(e, UnaryOp) and e.op == "neg":
            return visit(e.operand)
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            p = _ratio_fold_const(e.right)
            base = _get_flat_index(e.left, model)
            if base is not None and p is not None and p.is_integer() and 2 <= int(p) <= 4:
                indices.extend([base] * int(p))
                return True
        if isinstance(e, BinaryOp) and e.op == "/":
            d = _ratio_fold_const(e.right)
            if d is not None and d != 0.0:
                return visit(e.left)
            return False
        return False

    if not visit(expr):
        return None
    return indices


def extract_ratio_of_products(expr: Expression, model: Model) -> tuple[list[int], list[int]] | None:
    """Recognize ``(c·Πx_i)/(Πy_j)`` and return ``(num_indices, den_indices)``.

    Both numerator and denominator must reduce to a product of bounded original
    variables (constant scale factors folded away); the denominator must contain
    at least one variable (a constant denominator is plain scaling). Returns
    ``None`` for anything else (e.g. a transcendental or additive operand).
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "/"):
        return None
    num = _collect_ratio_product_vars(expr.left, model)
    if not num:
        return None
    den = _collect_ratio_product_vars(expr.right, model)
    if not den:
        return None
    return num, den


def extract_reciprocal_power(expr: Expression, model: Model) -> tuple[int, float, float] | None:
    """Recognize ``expr`` as ``c / (x**p)`` → ``(flat_idx, -p, c)``.

    Returns ``(flat_idx, exponent, coeff)`` such that ``expr == coeff *
    x_flat_idx ** exponent`` with a negative ``exponent``, or ``None``. Only a
    constant numerator over a single-variable power denominator is matched (the
    ``1/(x**3 * sqrt(x))`` shape in nvs08); a non-constant numerator or a
    multi-variable / scaled denominator returns ``None``.
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "/"):
        return None
    if not isinstance(expr.left, Constant):
        return None
    denom = extract_single_var_power(expr.right, model)
    if denom is None:
        return None
    flat_idx, exponent = denom
    if exponent <= 0.0:
        return None
    return (flat_idx, -exponent, float(expr.left.value))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def _classify_recursion_headroom(model: Model) -> int:
    """Recursion-limit headroom the Python classifier walk may need on *model*.

    Sized off the *deepest single expression*, not the constraint count: the
    hazard is depth concentrated in one giant body (issues #266/#271 hit the
    same shape).  Returns 0 when the default limit is safe; the estimate is a
    deliberate over-approximation (depth is bounded by node count) and only
    decides whether the deep-stack path engages, never what gets classified.
    """
    # Deferred: ``factorable_reform`` imports from this module at import time.
    from .factorable_reform import _DEEP_RECURSION_SIZE_GATE, _max_expr_node_count

    size = _max_expr_node_count(model)
    if size <= _DEEP_RECURSION_SIZE_GATE:
        return 0
    # ``distribute_products`` enters a couple of frames per expression node along
    # the deepest path, and ``_classify_node`` walks the rebuilt tree after it;
    # cushion the surrounding stack and cap it as the sibling walks do.
    return min(2000 + 6 * size, 600_000)


def classify_nonlinear_terms(model: Model) -> NonlinearTerms:
    """Walk the model's expression DAG and catalog nonlinear term structure.

    Uses the Rust expression-arena classifier for polynomial/product models when
    available, falling back to the Python implementation for unsupported models
    and for cases that need concrete ``general_nl`` expression objects.

    Which route ran is counted and logged (#1343): read it with
    :func:`classifier_route_counts`, and :func:`classifier_route_details` for the
    exception that caused a raising route. Every fallback is a named ``ROUTE_*``
    reason rather than a silent ``None``.

    The Python fallback (``distribute_products`` → ``_classify_node``) recurses
    per expression node, so a deep body blew the default 1000-frame limit and
    raised ``RecursionError``.  Callers treat that as "no reformulation
    available" rather than as the bug it is, so a model with a 53k-node body
    silently lost its whole term catalog.  Run the walk with size-scaled
    headroom on a large stack, reusing the runner proven for the convexity and
    factorable walks (issues #266/#271).
    """

    def _classify() -> tuple[NonlinearTerms, _RustAttempt]:
        if _contains_expandable_square(model):
            attempt = _RustAttempt(None, ROUTE_PY_EXPANDABLE_SQUARE)
        else:
            attempt = _classify_nonlinear_terms_rust(model)
            if attempt.terms is not None:
                return attempt.terms, attempt
        return _classify_nonlinear_terms_python(model), attempt

    from .convexity.rules import _run_with_deep_recursion

    # The route is returned rather than published from inside ``_classify``: on
    # the deep path that closure runs on a worker thread whose context copy is
    # discarded, so recording there would lose exactly the deep models (#1343).
    terms, attempt = _run_with_deep_recursion(
        _classify, depth_need=_classify_recursion_headroom(model)
    )
    _record_route(attempt)
    return terms


def _classify_nonlinear_terms_rust(model: Model) -> _RustAttempt:
    """Attempt the Rust fast path; on decline, name the route that declined it.

    Returns an :class:`_RustAttempt` whose ``terms`` is ``None`` for every
    decline. The exception arms record ``repr(exc)`` as ``detail`` instead of
    discarding it — a Rust failure and a Rust success must not look alike
    (#1343).
    """
    # #1520: narrowed. An unbuilt extension is the one failure this arm names
    # (``ROUTE_PY_IMPORT_FAILED``); anything else raised by the import is a defect.
    try:
        from discopt._rust import model_to_repr
    except ImportError as exc:
        return _RustAttempt(None, ROUTE_PY_IMPORT_FAILED, repr(exc))

    # Named ``model_repr``, not ``repr``: assigning the builtin's name makes it a
    # local for the WHOLE function, so the ``repr(exc)`` calls in these arms would
    # raise UnboundLocalError or call the ModelRepr object instead (#1343).
    try:
        model_repr = model_to_repr(model)
        payload = model_repr.classify_nonlinear_terms()
    except Exception as exc:
        return _RustAttempt(None, ROUTE_PY_CLASSIFY_RAISED, repr(exc))

    # The public API exposes the actual Python expression objects for general_nl.
    # The Rust arena sees only node ids, so keep those models on the Python path.
    if int(payload.get("general_nl_count", 0)) != 0:
        return _RustAttempt(None, ROUTE_PY_GENERAL_NL_OBJECTS)

    terms = _terms_from_rust_payload(payload)

    # Cross-check against the authoritative degree analysis. The Rust term
    # classifier has blind spots — a power/product over a *non-variable* base
    # (e.g. fac2's ``(x36+…+x41)**2.5``) is not categorised and reports zero
    # general_nl. If the model is provably *not* linear (objective or some
    # constraint has degree > 1 per ``max_degree``) yet the payload caught no
    # terms at all, the classification is incomplete: defer to the thorough
    # Python walk, which records the term in ``general_nl`` so the relaxation
    # builder and the simplex engine guard both see it.
    if not _terms_are_empty(terms):
        return _RustAttempt(terms, ROUTE_RUST)
    try:
        fully_linear = model_repr.is_objective_linear() and all(
            model_repr.is_constraint_linear(i) for i in range(model_repr.n_constraints)
        )
    except Exception as exc:
        return _RustAttempt(None, ROUTE_PY_DEGREE_CHECK_RAISED, repr(exc))
    if not fully_linear:
        # Nonlinear but nothing catalogued → the Python walk records it.
        return _RustAttempt(None, ROUTE_PY_CATALOG_INCOMPLETE)
    return _RustAttempt(terms, ROUTE_RUST)


def _terms_are_empty(terms: NonlinearTerms) -> bool:
    """True if no nonlinear term of any category was recorded."""
    return not (
        terms.bilinear
        or terms.trilinear
        or terms.multilinear
        or terms.monomial
        or terms.fractional_power
        or terms.bilinear_with_fp
        or terms.ratio_of_products
        or terms.general_nl
    )


def _terms_from_rust_payload(payload: dict[str, Any]) -> NonlinearTerms:
    """Convert the PyO3 classifier payload into the public dataclass."""
    incidence_payload = payload.get("term_incidence", {})
    return NonlinearTerms(
        bilinear=[(int(i), int(j)) for i, j in payload.get("bilinear", [])],
        trilinear=[(int(i), int(j), int(k)) for i, j, k in payload.get("trilinear", [])],
        multilinear=[tuple(int(idx) for idx in term) for term in payload.get("multilinear", [])],
        monomial=[(int(var_idx), int(exp)) for var_idx, exp in payload.get("monomial", [])],
        general_nl=[],
        term_incidence={
            int(var_idx): {int(term_idx) for term_idx in term_ids}
            for var_idx, term_ids in incidence_payload.items()
        },
        partition_candidates=[int(var_idx) for var_idx in payload.get("partition_candidates", [])],
    )


def _classify_nonlinear_terms_python(model: Model) -> NonlinearTerms:
    """Walk the model's expression DAG and catalog nonlinear term structure.

    Scans all constraints and the objective.  Each unique bilinear/trilinear/monomial
    pattern is recorded at most once (deduplicated by sorted variable index tuple).

    Parameters
    ----------
    model : Model
        A discopt Model with objective and constraints set.

    Returns
    -------
    NonlinearTerms
        Catalog of nonlinear terms ready for AMP partitioning.
    """
    result = NonlinearTerms()

    # Track seen terms to avoid duplicates
    seen_bilinear: set[tuple[int, int]] = set()
    seen_trilinear: set[tuple[int, int, int]] = set()
    seen_multilinear: set[tuple[int, ...]] = set()
    seen_monomial: set[tuple[int, int]] = set()
    # #1591: variable-free memo for the array-nonlinear test (pure structural).
    vf_memo: dict[int, bool] = {}
    seen_fractional: set[tuple[int, float]] = set()
    seen_bilinear_fp: set[tuple[int, tuple[int, float]]] = set()

    def _next_product_term_idx() -> int:
        return len(result.bilinear) + len(result.trilinear) + len(result.multilinear)

    def _record_bilinear(i: int, j: int) -> None:
        key = (min(i, j), max(i, j))
        if key not in seen_bilinear:
            seen_bilinear.add(key)
            term_idx = _next_product_term_idx()
            result.bilinear.append(key)
            # Update term incidence
            for v in key:
                result.term_incidence.setdefault(v, set()).add(term_idx)

    def _record_trilinear(i: int, j: int, k: int) -> None:
        a, b, c = sorted((i, j, k))
        key = (a, b, c)
        if key not in seen_trilinear:
            seen_trilinear.add(key)
            term_idx = _next_product_term_idx()
            result.trilinear.append(key)
            for v in key:
                result.term_incidence.setdefault(v, set()).add(term_idx)

    def _record_multilinear(indices: list[int]) -> None:
        key = tuple(sorted(indices))
        if len(key) < 4:
            raise ValueError("multilinear terms require at least four variables")
        if key not in seen_multilinear:
            seen_multilinear.add(key)
            term_idx = _next_product_term_idx()
            result.multilinear.append(key)
            for v in key:
                result.term_incidence.setdefault(v, set()).add(term_idx)

    def _record_monomial(var_idx: int, exp: int) -> None:
        key = (var_idx, exp)
        if key not in seen_monomial:
            seen_monomial.add(key)
            result.monomial.append(key)

    def _record_fractional_power(var_idx: int, exp: float) -> None:
        key = (var_idx, float(exp))
        if key not in seen_fractional:
            seen_fractional.add(key)
            result.fractional_power.append(key)

    def _record_bilinear_with_fp(var_idx: int, fp: tuple[int, float]) -> None:
        fp_norm = (fp[0], float(fp[1]))
        key = (var_idx, fp_norm)
        if key not in seen_bilinear_fp:
            seen_bilinear_fp.add(key)
            result.bilinear_with_fp.append(key)
        _record_fractional_power(*fp_norm)

    def _record_product_indices(indices: list[int]) -> None:
        """Record a distinct-variable product as a bilinear/trilinear/multilinear
        term so it receives a McCormick envelope. Repeated-factor products (powers)
        are skipped here; their variables still become partition candidates via the
        ``ratio_of_products`` record."""
        unique = list(dict.fromkeys(indices))
        if len(unique) != len(indices):
            return
        if len(unique) == 2:
            _record_bilinear(unique[0], unique[1])
        elif len(unique) == 3:
            _record_trilinear(unique[0], unique[1], unique[2])
        elif len(unique) >= 4:
            _record_multilinear(unique)

    def _classify_array_nonlinear(expr: Expression) -> None:
        """Classify an element-wise-nonlinear array node by its scalar elements (#1591).

        ``x @ Q @ x`` is the contraction ``sum_k (x @ Q)[k] * x[k]`` and
        ``x * (Q @ x)`` is the vector of ``x[k] * sum_i Q[k, i] x[i]``; only once
        expanded (and distributed) do their bilinear/monomial terms exist. Before
        this the ``@`` arm recursed into its operands -- "A @ x is linear if A is
        constant" -- and catalogued *nothing*, so the LP relaxer saw no relaxable
        nonlinearity and the solve fell to the alphaBB route and stopped at the
        root. The classification of a reduction is the union over its operand's
        elements, so expanding the node itself covers every enclosing
        ``sum``/index/function.
        """
        if isinstance(expr, MatMulExpression) and static_shape(expr) == ():
            contraction = scalar_matmul_contraction(expr)
            elems = None if contraction is None else [contraction]
        else:
            elems = scalar_elements(expr)
            if elems is not None and len(elems) == 1 and elems[0] is expr:
                elems = None  # identity: no static expansion of this node
        if elems is None:
            # No static expansion (unknown shape / over the cap): keep the node as
            # one opaque nonlinear term, so nothing downstream reads it as linear.
            result.general_nl.append(expr)
            _classify_node(expr.left)  # type: ignore[attr-defined]
            _classify_node(expr.right)  # type: ignore[attr-defined]
            return
        for elem in elems:
            _classify_node(distribute_products(_rebalance_additive(elem)))

    def _classify_node(expr: Expression) -> None:
        """Recursively classify all nonlinear nodes in the expression tree."""
        if isinstance(expr, Constant):
            return

        if isinstance(expr, Variable):
            return  # bare variable — linear

        if _is_array_nonlinear(expr, vf_memo):
            _classify_array_nonlinear(expr)
            return

        if isinstance(expr, IndexExpression):
            # x[i] — linear leaf; recurse into base only if it's something unusual
            if not isinstance(expr.base, Variable):
                _classify_node(expr.base)
            return

        if isinstance(expr, BinaryOp):
            # ── Power: x**n ──
            if expr.op == "**":
                flat = _get_flat_index(expr.left, model)
                if flat is not None and isinstance(expr.right, Constant):
                    exp_val = float(expr.right.value)
                    if exp_val == int(exp_val) and int(exp_val) >= 2:
                        _record_monomial(flat, int(exp_val))
                        return
                    elif exp_val != 1.0:
                        # Non-integer (or negative-integer) exponent → fractional
                        # power.  Record both as a fractional_power term (so the
                        # MILP relaxation can lift it to an aux variable) and in
                        # general_nl (so legacy callers see the same term set).
                        _record_fractional_power(flat, exp_val)
                        result.general_nl.append(expr)
                        return
                # Power whose base is NOT a single variable (or whose exponent is
                # not constant). Anything other than a variable-free constant power
                # or a degree-1 passthrough is genuinely nonlinear and would be
                # SILENTLY DROPPED by the linear projection (extract_lp_data),
                # making the simplex engine certify a wrong 'optimal' — e.g. fac2's
                # ``(x36+…+x41)**2.5`` objective (carton7/#286 class). Flag it as
                # general_nl so the engine guard defers and the relaxation lifts it.
                exp_const = _ratio_fold_const(expr.right)
                whole_is_const = _ratio_fold_const(expr.left) is not None and exp_const is not None
                if whole_is_const:
                    return  # variable-free → folds to a constant
                if exp_const is not None and exp_const == 1.0:
                    _classify_node(expr.left)  # base**1 == base (linear iff base is)
                    return
                result.general_nl.append(expr)
                _classify_node(expr.left)
                _classify_node(expr.right)
                return

            # ── Multiplication: try product-tree decomposition ──
            if expr.op == "*":
                factors = _collect_product_factors(expr, model)
                if factors is not None:
                    unique_vars = list(dict.fromkeys(factors))  # preserve order, remove dups
                    n_unique = len(unique_vars)
                    counts = {v: factors.count(v) for v in unique_vars}
                    if n_unique == 1:
                        # x * x = x^2 → monomial
                        _record_monomial(unique_vars[0], counts[unique_vars[0]])
                        return
                    if any(c >= 2 for c in counts.values()):
                        # Mixed repeated-factor products such as x*x*y are not
                        # represented correctly by the current bilinear/trilinear
                        # relaxation pipeline. Keep the whole product in general_nl
                        # without also classifying subproducts from the same term.
                        result.general_nl.append(expr)
                        return
                    if n_unique == 2:
                        _record_bilinear(unique_vars[0], unique_vars[1])
                        return
                    elif n_unique == 3:
                        _record_trilinear(unique_vars[0], unique_vars[1], unique_vars[2])
                        return
                    else:
                        _record_multilinear(unique_vars)
                        return
                # Pure-variable decomposition failed.  Try the extended walk
                # which permits fractional powers as virtual factors.
                ext = _collect_extended_factors(expr, model)
                if ext is not None:
                    flat_facs, fp_facs = ext
                    unique_flat = list(dict.fromkeys(flat_facs))
                    if len(fp_facs) == 1 and len(flat_facs) == 1:
                        # Pattern: x * y^p  →  bilinear-with-fractional-power.
                        _record_bilinear_with_fp(flat_facs[0], fp_facs[0])
                        return
                    if len(fp_facs) == 1 and len(flat_facs) == 0:
                        # Pattern: c * y^p  →  pure fractional power.
                        _record_fractional_power(*fp_facs[0])
                        return
                    if len(fp_facs) == 0 and len(unique_flat) >= 1:
                        # Should have been caught by _collect_product_factors;
                        # falling through to general_nl is the safe choice.
                        pass
                    # Any other shape (multiple fp factors, fp × bilinear, …) is
                    # outside the supported relaxations: keep as general_nl and
                    # recurse so nested simple terms can still be classified.
                    result.general_nl.append(expr)
                    _classify_node(expr.left)
                    _classify_node(expr.right)
                    return
                # Product decomposition failed. A product of two non-constant
                # sub-expressions — e.g. (x+y)*(z+w) or (x+y)*z — is nonlinear and
                # would be silently dropped by the linear projection; flag it.
                # If either side folds to a constant the product is just linear
                # scaling, so only recurse (the existing behaviour).
                # #1614 C1: a *variable-free* factor -- a literal constant OR a
                # ``Parameter`` (fixed for the solve) -- is linear scaling, the same
                # as a constant. ``_ratio_fold_const`` folds only literals, so
                # ``p*(x+y)`` was flagged ``general_nl`` and its MILP left HiGHS for
                # the native tree while the algebraically equal ``p*x+p*y`` did not.
                left_const = _is_variable_free(expr.left, vf_memo)
                right_const = _is_variable_free(expr.right, vf_memo)
                if not left_const and not right_const:
                    result.general_nl.append(expr)
                _classify_node(expr.left)
                _classify_node(expr.right)
                return

            # ── Other binary ops: +, -, / ──
            if expr.op in ("+", "-"):
                _classify_node(expr.left)
                _classify_node(expr.right)
                return

            if expr.op == "/":
                # x / c where c is constant → linear scaling
                if isinstance(expr.right, Constant):
                    _classify_node(expr.left)
                    return
                # c / (x**p)  →  fractional power x**-p (e.g. 1/(x**3*sqrt(x))
                # in nvs08 → x**-3.5). Record it so the MILP relaxation lifts it
                # to an aux column instead of dropping the whole constraint.
                recip = extract_reciprocal_power(expr, model)
                if recip is not None:
                    flat_idx, neg_exp, _coeff = recip
                    _record_fractional_power(flat_idx, neg_exp)
                    result.general_nl.append(expr)
                    return
                # (c·Πx)/(Πy) → ratio of products (issue #185). Register the
                # numerator and denominator products so they receive McCormick
                # envelopes and their variables become partition candidates; the
                # MILP relaxation lifts the quotient via the r·q = m identity.
                ratio = extract_ratio_of_products(expr, model)
                if ratio is not None:
                    num_idx, den_idx = ratio
                    _record_product_indices(num_idx)
                    _record_product_indices(den_idx)
                    result.ratio_of_products.append(
                        (
                            tuple(sorted(dict.fromkeys(num_idx))),
                            tuple(sorted(dict.fromkeys(den_idx))),
                        )
                    )
                    result.general_nl.append(expr)
                    return
                # c / x or x / y → general nonlinear
                result.general_nl.append(expr)
                return

            # Fallthrough: recurse
            _classify_node(expr.left)
            _classify_node(expr.right)
            return

        if isinstance(expr, UnaryOp):
            if expr.op == "neg":
                _classify_node(expr.operand)
                return
            # abs → nonlinear
            result.general_nl.append(expr)
            return

        if isinstance(expr, FunctionCall):
            # All named functions are considered nonlinear (transcendental)
            # sin, cos, exp, log, sqrt, tan, etc.
            result.general_nl.append(expr)
            # Recurse into arguments (they might contain bilinear sub-expressions)
            for arg in expr.args:
                _classify_node(arg)
            return

        if isinstance(expr, SumExpression):
            _classify_node(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                _classify_node(term)
            return

        if isinstance(expr, MatMulExpression):
            # One side is variable-free here (the both-sides case was expanded by
            # ``_classify_array_nonlinear`` above), so ``A @ v`` is linear in ``v``
            # -- unless ``v`` itself is nonlinear, which the recursion classifies.
            _classify_node(expr.left)
            _classify_node(expr.right)
            return

    # ── Scan objective ──
    # Distribute multiplication over addition/subtraction first so that products
    # of the form ``y * (x^p - c)`` decompose into ``y*x^p - y*c``, exposing the
    # ``y * x^p`` bilinear-with-fractional-power pattern to classification.
    #
    # #1456: every body goes through ONE per-pass distribution budget
    # (``distribute_bodies``), not just the per-call one. Unchanged wherever the
    # model's total distribution fits it (every MINLPLib instance but johnall and
    # saa_2); there the oversized bodies stay undistributed, and an intact product
    # of non-constant factors is flagged ``general_nl`` below -- the same answer a
    # per-call-truncated product already gets.
    bodies: list[Expression] = []
    if model._objective is not None:
        bodies.append(model._objective.expression)

    # ── Scan constraints ──
    # Array-valued bodies are expanded element-wise first (#981). A vectorized
    # product such as ``(-k) * X[:, 1:]`` is ONE expression node standing for a
    # whole matrix of scalar products; classified whole it matches no scalar
    # pattern and lands in ``general_nl``, so its variables never become
    # partition candidates and the spatial machinery sees no bilinear structure
    # at all. Expanded, each element is an ordinary bilinear term.
    for constraint in model._constraints:
        rows = scalar_elements(constraint.body)
        if rows is None:
            rows = [constraint.body]
        bodies.extend(rows)
    for dist in distribute_bodies(bodies, pass_name="classify_nonlinear_terms"):
        _classify_node(dist)

    # ── Build partition_candidates ──
    # Variables that appear in product terms (not just monomials, since x^2 is
    # convex and handled by alphaBB/direct secant).
    candidates: set[int] = set()
    for i, j in result.bilinear:
        candidates.add(i)
        candidates.add(j)
    for i, j, k in result.trilinear:
        candidates.add(i)
        candidates.add(j)
        candidates.add(k)
    for term in result.multilinear:
        candidates.update(term)
    # Bilinear-with-fractional-power lifts the fp into an aux column, but the
    # underlying base variable still needs domain partitioning to tighten the
    # secant/tangent envelopes on a = x^p.
    for lin_idx, (fp_base, _exp) in result.bilinear_with_fp:
        candidates.add(lin_idx)
        candidates.add(fp_base)
    for fp_base, _exp in result.fractional_power:
        candidates.add(fp_base)
    # Ratio-of-products variables (numerator and denominator) drive the
    # linear-fractional envelope; partitioning them tightens it (issue #185).
    for num_vars, den_vars in result.ratio_of_products:
        candidates.update(num_vars)
        candidates.update(den_vars)
    result.partition_candidates = sorted(candidates)

    return result
