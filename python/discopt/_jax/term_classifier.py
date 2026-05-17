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
    general_nl: list[Expression] = field(default_factory=list)
    term_incidence: dict[_VarIdx, set[int]] = field(default_factory=dict)
    partition_candidates: list[_VarIdx] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers: flat index extraction
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting flat index of a variable in the stacked x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


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
    """Return True if Python classification should expand a non-leaf square."""

    def visit(expr: Expression) -> bool:
        if isinstance(expr, BinaryOp):
            if (
                expr.op == "**"
                and isinstance(expr.right, Constant)
                and float(expr.right.value) == 2.0
                and _get_flat_index(expr.left, model) is None
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


def distribute_products(expr: Expression) -> Expression:
    """Recursively distribute multiplication over addition/subtraction.

    ``(a + b) * c`` → ``a*c + b*c``;  ``c * (a - b)`` → ``c*a - c*b``.
    ``(a + b)^2`` → ``(a + b) * (a + b)`` before distribution.
    Applied bottom-up so nested distributions resolve.  Other expression
    types are returned with operator-tree shape preserved structurally.
    """
    if isinstance(expr, BinaryOp):
        left = distribute_products(expr.left)
        right = distribute_products(expr.right)
        if expr.op == "**" and isinstance(right, Constant) and float(right.value) == 2.0:
            return distribute_products(BinaryOp("*", left, left))
        if expr.op == "*":
            if isinstance(right, BinaryOp) and right.op in ("+", "-"):
                return BinaryOp(
                    right.op,
                    distribute_products(BinaryOp("*", left, right.left)),
                    distribute_products(BinaryOp("*", left, right.right)),
                )
            if isinstance(left, BinaryOp) and left.op in ("+", "-"):
                return BinaryOp(
                    left.op,
                    distribute_products(BinaryOp("*", left.left, right)),
                    distribute_products(BinaryOp("*", left.right, right)),
                )
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, distribute_products(expr.operand))
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
        payload = model_to_repr(model).classify_nonlinear_terms()
    except Exception:
        return None

    # The public API exposes the actual Python expression objects for general_nl.
    # The Rust arena sees only node ids, so keep those models on the Python path.
    if int(payload.get("general_nl_count", 0)) != 0:
        return None

    return _terms_from_rust_payload(payload)


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
                # Recurse for complex bases
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
                # If product decomposition failed, recurse on children
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
    result.partition_candidates = sorted(candidates)

    return result
