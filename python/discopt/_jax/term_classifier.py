"""
Nonlinear Term Classifier for AMP (Adaptive Multivariate Partitioning).

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

Theory references:
  - Nagarajan et al., CP 2016: http://harshangrjn.github.io/pdf/CP_2016.pdf
  - Nagarajan et al., JOGO 2018: http://harshangrjn.github.io/pdf/JOGO_2018.pdf
  - Alpine.jl operators.jl / nlexpr.jl
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import index as operator_index
from typing import Any

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

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


def _as_scalar_index(value: Any) -> int | None:
    """Return a Python integer index, or None for slices and non-scalars."""
    try:
        return operator_index(value)
    except TypeError:
        return None


def _tuple_to_flat_index(indices: Sequence[int], shape: Sequence[int]) -> int | None:
    """Flatten a scalar multidimensional index in row-major order."""
    if len(indices) != len(shape):
        return None

    flat = 0
    stride = 1
    for idx, dim in zip(reversed(indices), reversed(shape)):
        flat += idx * stride
        stride *= dim
    return flat


def _get_flat_index(expr: Expression, model: Model) -> int | None:
    """Return the flat variable index for a scalar Variable or IndexExpression.

    Returns None if the expression is not a scalar variable reference.
    """
    if isinstance(expr, Variable):
        if expr.size == 1:
            return _compute_var_offset(expr, model)
        return None  # multi-element variable without index — can't reduce to scalar
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base_off = _compute_var_offset(expr.base, model)
        idx = expr.index
        scalar_idx = _as_scalar_index(idx)
        if scalar_idx is not None:
            return base_off + scalar_idx
        if isinstance(idx, tuple):
            scalar_indices = []
            for item in idx:
                item_idx = _as_scalar_index(item)
                if item_idx is None:
                    return None
                scalar_indices.append(item_idx)
            flat = _tuple_to_flat_index(scalar_indices, expr.base.shape)
            if flat is not None:
                return base_off + flat
    return None


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
    """

    def _is_additive_composite(e: Expression) -> bool:
        """A ``+``/``-`` node whose distribution can expose product cross-terms."""
        return isinstance(e, BinaryOp) and e.op in ("+", "-")

    def visit(expr: Expression) -> bool:
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
            return visit(expr.left) or visit(expr.right)
        if isinstance(expr, UnaryOp):
            return visit(expr.operand)
        if isinstance(expr, FunctionCall):
            return any(visit(arg) for arg in expr.args)
        if isinstance(expr, IndexExpression):
            return not isinstance(expr.base, Variable) and visit(expr.base)
        if isinstance(expr, SumExpression):
            return visit(expr.operand)
        if isinstance(expr, SumOverExpression):
            return any(visit(term) for term in expr.terms)
        if isinstance(expr, MatMulExpression):
            return visit(expr.left) or visit(expr.right)
        return False

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


def distribute_products(
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
                base = distribute_products(expr.left, protected_squares)
                expanded = _distribute_power_over_product(base, n_int)
                if expanded is not None:
                    return expanded
            if exp_val == 2.0:
                left = distribute_products(expr.left, protected_squares)
                return _distribute_mul(left, left)
        left = distribute_products(expr.left, protected_squares)
        right = distribute_products(expr.right, protected_squares)
        if expr.op == "*":
            return _distribute_mul(left, right)
        # Preserve node identity when nothing distributed, so id()-keyed maps
        # (e.g. composite/univariate aux columns) still match the rebuilt tree.
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = distribute_products(expr.operand, protected_squares)
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


def classify_nonlinear_terms(model: Model) -> NonlinearTerms:
    """Walk the model's expression DAG and catalog nonlinear term structure.

    Uses the Rust expression-arena classifier for polynomial/product models when
    available, falling back to the Python implementation for unsupported models
    and for cases that need concrete ``general_nl`` expression objects.
    """
    if not _contains_expandable_square(model):
        rust_terms = _classify_nonlinear_terms_rust(model)
        if rust_terms is not None:
            return rust_terms
    return _classify_nonlinear_terms_python(model)


def _classify_nonlinear_terms_rust(model: Model) -> NonlinearTerms | None:
    """Return Rust-classified terms when the fast path can preserve the API."""
    try:
        from discopt._rust import model_to_repr
    except Exception:
        return None

    try:
        repr = model_to_repr(model)
        payload = repr.classify_nonlinear_terms()
    except Exception:
        return None

    # The public API exposes the actual Python expression objects for general_nl.
    # The Rust arena sees only node ids, so keep those models on the Python path.
    if int(payload.get("general_nl_count", 0)) != 0:
        return None

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
        return terms
    try:
        fully_linear = repr.is_objective_linear() and all(
            repr.is_constraint_linear(i) for i in range(repr.n_constraints)
        )
    except Exception:
        return None
    if not fully_linear:
        return None  # nonlinear but nothing catalogued → Python path
    return terms


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

    def _classify_node(expr: Expression) -> None:
        """Recursively classify all nonlinear nodes in the expression tree."""
        if isinstance(expr, Constant):
            return

        if isinstance(expr, Variable):
            return  # bare variable — linear

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
                left_const = _ratio_fold_const(expr.left) is not None
                right_const = _ratio_fold_const(expr.right) is not None
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
            # A @ x is linear if A is constant — recurse for safety
            _classify_node(expr.left)
            _classify_node(expr.right)
            return

    # ── Scan objective ──
    # Distribute multiplication over addition/subtraction first so that products
    # of the form ``y * (x^p - c)`` decompose into ``y*x^p - y*c``, exposing the
    # ``y * x^p`` bilinear-with-fractional-power pattern to classification.
    if model._objective is not None:
        _classify_node(distribute_products(model._objective.expression))

    # ── Scan constraints ──
    for constraint in model._constraints:
        _classify_node(distribute_products(constraint.body))

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
