"""
Nonlinear Term Classifier for AMP (Adaptive Multivariate Partitioning).

Walks the expression DAG of a Model and catalogs nonlinear term structure:
  - bilinear terms:   x_i * x_j  (two distinct continuous variables)
  - trilinear terms:  x_i * x_j * x_k  (three distinct continuous variables)
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

from dataclasses import dataclass, field

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
    monomial : list of (int, int)
        Each entry is (var_idx, exponent) for x_i^n, n integer ≥ 2.
    general_nl : list of Expression
        Nonlinear expression nodes that are neither bilinear, trilinear, nor monomial
        (e.g., sin, cos, exp, log, sqrt, tan, abs).
    term_incidence : dict[int, set[int]]
        Maps flat variable index → set of term indices (into the combined bilinear +
        trilinear list) that the variable appears in.  Used for vertex-cover computation.
    partition_candidates : list[int]
        Sorted list of flat variable indices appearing in any bilinear or trilinear term.
        These are the candidates for domain partitioning in AMP.
        (Monomials are convex/treated separately; general_nl may also be candidates
        but are currently excluded from partitioning as AMP focuses on polynomial terms.)
    """

    bilinear: list[tuple[_VarIdx, _VarIdx]] = field(default_factory=list)
    trilinear: list[tuple[_VarIdx, _VarIdx, _VarIdx]] = field(default_factory=list)
    monomial: list[tuple[_VarIdx, int]] = field(default_factory=list)
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
        if isinstance(idx, int):
            return base_off + idx
        if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            return base_off + idx[0]
    return None


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


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_nonlinear_terms(model: Model) -> NonlinearTerms:
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
    seen_monomial: set[tuple[int, int]] = set()

    def _record_bilinear(i: int, j: int) -> None:
        key = (min(i, j), max(i, j))
        if key not in seen_bilinear:
            seen_bilinear.add(key)
            term_idx = len(result.bilinear) + len(result.trilinear)
            result.bilinear.append(key)
            # Update term incidence
            for v in key:
                result.term_incidence.setdefault(v, set()).add(term_idx)

    def _record_trilinear(i: int, j: int, k: int) -> None:
        key = tuple(sorted([i, j, k]))
        if key not in seen_trilinear:
            seen_trilinear.add(key)
            term_idx = len(result.bilinear) + len(result.trilinear)
            result.trilinear.append(key)  # type: ignore[arg-type]
            for v in key:
                result.term_incidence.setdefault(v, set()).add(term_idx)

    def _record_monomial(var_idx: int, exp: int) -> None:
        key = (var_idx, exp)
        if key not in seen_monomial:
            seen_monomial.add(key)
            result.monomial.append(key)

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
                        # Non-integer or non-trivial exponent → general nonlinear
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
                    if n_unique == 1:
                        # x * x = x^2 → monomial
                        _record_monomial(unique_vars[0], factors.count(unique_vars[0]))
                        return
                    elif n_unique == 2:
                        # Check if any var appears twice → monomial
                        counts = {v: factors.count(v) for v in unique_vars}
                        if any(c >= 2 for c in counts.values()):
                            for v, c in counts.items():
                                if c >= 2:
                                    _record_monomial(v, c)
                            # The other var is a separate multiplier — treat whole as bilinear-like
                            # e.g. x^2 * y: still bilinear between (x_squared, y)
                            # For simplicity, record as bilinear
                            _record_bilinear(unique_vars[0], unique_vars[1])
                        else:
                            _record_bilinear(unique_vars[0], unique_vars[1])
                        return
                    elif n_unique == 3:
                        _record_trilinear(unique_vars[0], unique_vars[1], unique_vars[2])
                        return
                    else:
                        # Higher-order multilinear — decompose into bilinear pairs
                        # (AMP handles multilinear via repeated bilinear decomposition)
                        for ii in range(len(unique_vars)):
                            for jj in range(ii + 1, len(unique_vars)):
                                _record_bilinear(unique_vars[ii], unique_vars[jj])
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
    if model._objective is not None:
        _classify_node(model._objective.expression)

    # ── Scan constraints ──
    for constraint in model._constraints:
        _classify_node(constraint.body)

    # ── Build partition_candidates ──
    # Variables that appear in bilinear or trilinear terms (not just monomials,
    # since x^2 is convex and handled by alphaBB/direct secant).
    candidates: set[int] = set()
    for i, j in result.bilinear:
        candidates.add(i)
        candidates.add(j)
    for i, j, k in result.trilinear:
        candidates.add(i)
        candidates.add(j)
        candidates.add(k)
    result.partition_candidates = sorted(candidates)

    return result
