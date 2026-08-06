"""
Problem classification and standard-form extraction for LP, QP, QCP, MILP,
MIQP, MIQCP, NLP, MINLP.

Uses existing Rust structure detection (is_linear, is_quadratic) via PyO3 bindings
to classify problems, then extracts standard-form data using the JAX DAG compiler.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    # Annotation-only: ``LPData``/``QPData`` are typed as jnp arrays to match the
    # JAX differentiation consumers (``differentiable_qp`` etc.); at runtime they hold numpy from
    # the JAX-free extractors. ``from __future__ import annotations`` keeps these
    # strings, so no JAX import happens at module load.
    import jax.numpy as jnp

# NOTE: ``jax`` is imported lazily inside the ``extract_*`` data functions that
# build jnp arrays. ``classify_problem`` and the dataclasses are purely
# structural (annotations are strings under ``from __future__ import
# annotations``), so importing this module — done on every ``Model.solve`` to
# route the problem class — does not pull in JAX. That keeps LP/MILP/MIQP solves
# free of JAX/XLA cold-start.
from discopt._flat_index import resolve_scalar_slot
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)


class ProblemClass(Enum):
    """Classification of an optimization problem."""

    LP = "lp"  # linear obj + linear constraints + all continuous
    QP = "qp"  # ≤quadratic obj + linear constraints + all continuous
    QCP = "qcp"  # linear obj + at least one quadratic constraint + all continuous
    QCQP = "qcqp"  # ≤quadratic obj + at least one quadratic constraint + all continuous
    MILP = "milp"  # linear obj + linear constraints + has integer/binary
    MIQP = "miqp"  # ≤quadratic obj + linear constraints + has integer/binary
    MIQCP = "miqcp"  # linear obj + at least one quadratic constraint + has integer/binary
    MIQCQP = "miqcqp"  # ≤quadratic obj + quadratic constraints + has integer/binary
    NLP = "nlp"  # general nonlinear + all continuous
    MINLP = "minlp"  # general nonlinear + has integer/binary


def classify_problem(model: Model) -> ProblemClass:
    """Classify a model into LP, QP, QCP, MILP, MIQP, MIQCP, NLP, or MINLP.

    Uses Rust structure detection for degree analysis of the objective
    and constraints. Falls back to NLP/MINLP if Rust bindings unavailable.

    Args:
        model: A discopt Model with objective and constraints.

    Returns:
        ProblemClass enum value.
    """
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)

    try:
        from discopt._rust import model_to_repr

        _builder = getattr(model, "_builder", None)
        repr = model_to_repr(model, _builder)
        obj_linear = repr.is_objective_linear()
        obj_quadratic = repr.is_objective_quadratic()
        all_constraints_linear = all(
            repr.is_constraint_linear(i) for i in range(repr.n_constraints)
        )
        if hasattr(repr, "is_constraint_quadratic"):
            all_constraints_quadratic = all(
                repr.is_constraint_quadratic(i) for i in range(repr.n_constraints)
            )
        else:
            all_constraints_quadratic = all_constraints_linear
    except Exception as exc:  # noqa: BLE001 - NLP/MINLP is the always-valid fallback class
        # Not merely an optimization: misrouting an LP/QP to the MINLP path is the
        # difference between the fast family and full spatial B&B, so a silent
        # degradation here reads as "the fast family didn't trigger".
        logger.debug(
            "Rust structure detection unavailable, classifying as NLP/MINLP: %s: %s",
            type(exc).__name__,
            exc,
        )
        return ProblemClass.MINLP if has_integer else ProblemClass.NLP

    if all_constraints_linear:
        if obj_linear:
            return ProblemClass.MILP if has_integer else ProblemClass.LP
        if obj_quadratic:
            return ProblemClass.MIQP if has_integer else ProblemClass.QP

    if all_constraints_quadratic:
        if obj_linear:
            return ProblemClass.MIQCP if has_integer else ProblemClass.QCP
        if obj_quadratic:
            return ProblemClass.MIQCQP if has_integer else ProblemClass.QCQP

    return ProblemClass.MINLP if has_integer else ProblemClass.NLP


# Dense-Q budget for QP extraction (#863). Above this the extractor emits a sparse
# Q instead: a wide model with a narrow objective (watercontamination0202 is 106,711
# variables whose objective touches 101) cannot hold (n, n) float64 at all -- that
# is 91 GB. 256 MB corresponds to n ~= 5,657.
_QP_DENSE_Q_MAX_BYTES = 256 * 1024 * 1024

# Dense-constraint-matrix budget, the exact counterpart for A_eq / A_ub (#863).
# ``watercontamination0202`` is 107,209 rows x 106,711 columns; (m, n) float64 there
# is 91.5 GB, an equal-sized wall to the 91 GB dense Q above. Above this budget the
# extractors emit scipy CSR and consumers densify through ``dense_A()``.
_DENSE_A_MAX_BYTES = 256 * 1024 * 1024


def _sp_issparse(x) -> bool:
    import scipy.sparse as _sp

    return bool(_sp.issparse(x))


class LPData(NamedTuple):
    """Standard-form LP data: min c'x + d s.t. A_eq x = b_eq, x_l <= x <= x_u."""

    c: jnp.ndarray  # (n,) objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


class QPData(NamedTuple):
    """Standard-form QP: min 0.5 x'Qx + c'x + d s.t. A_eq x = b_eq, bounds."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def dense_Q(Q) -> np.ndarray:
    """Materialise a ``QPData.Q`` as a dense ``float64`` array.

    ``Q`` may be a dense array or a scipy sparse matrix (#863: a wide model with a
    narrow objective — ``watercontamination0202`` is 106,711 variables whose
    objective touches 101 — cannot hold a dense ``(n, n)`` ``Q``; that is 91 GB).

    Returns numpy, which is what these fields actually hold at runtime (the module
    header notes ``QPData`` is *annotated* for the JAX differentiation consumers but
    populated by the JAX-free extractors). The call sites that feed POUNCE's
    ``solve_qp_kkt`` cast at the boundary rather than have this function lie about
    its type.

    Every consumer must go through this rather than calling ``np.asarray`` on ``Q``
    directly. ``np.asarray`` on a scipy sparse matrix does **not** raise — it
    returns a 0-d object array wrapping the matrix — so a missed call site would
    silently feed garbage into a solver instead of failing loudly. That is the whole
    reason this helper exists.
    """
    try:
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            return np.asarray(Q.toarray(), dtype=np.float64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    # Inspect BEFORE coercing to float64: np.asarray(obj_array, dtype=float) raises
    # a bare ValueError, which hides what actually went wrong.
    arr = np.asarray(Q)
    if arr.dtype == object or arr.ndim != 2:
        raise TypeError(
            f"QPData.Q densified to dtype={arr.dtype} ndim={arr.ndim}; a sparse "
            "matrix probably reached np.asarray without going through dense_Q()"
        )
    return np.asarray(arr, dtype=np.float64)


def dense_A(A) -> np.ndarray:
    """Materialise a constraint matrix as a dense ``float64`` 2-D array.

    Covers ``LPData.A_eq``, ``QPData.A_eq`` and ``QCPData.A_ub`` / ``A_eq``. Each
    may be a dense array or a scipy sparse matrix (#863). The dense form is
    ``(m, n)``: on ``watercontamination0202`` — 106,711 variables, 107,209 rows —
    that is **91.5 GB**, an equal-sized wall to the 91 GB dense ``Q`` that the same
    issue removed, and it is why extraction on that instance never returns.

    The sibling of :func:`dense_Q`, and for the same reason. ``np.asarray`` on a
    scipy sparse matrix does **not** raise — it returns a 0-d object array wrapping
    the matrix — so a consumer that keeps calling ``np.asarray(lp_data.A_eq)``
    directly would silently feed garbage into a solver instead of failing. Every
    consumer must go through this, and it raises loudly (``TypeError`` naming the
    cause) if it ever yields an object array or a non-2-D result, which is the
    signature of a missed call site.

    Returns numpy, which is what these fields actually hold at runtime (the module
    header notes the data classes are *annotated* for the JAX consumers but
    populated by the JAX-free extractors); the sites that feed POUNCE's
    ``solve_qp_kkt`` cast at that boundary rather than have this function lie
    about its type.
    """
    try:
        import scipy.sparse as _sp

        if _sp.issparse(A):
            return np.asarray(A.toarray(), dtype=np.float64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    # Inspect BEFORE coercing to float64: np.asarray(obj_array, dtype=float) raises
    # a bare ValueError, which hides what actually went wrong.
    arr = np.asarray(A)
    if arr.dtype == object or arr.ndim != 2:
        raise TypeError(
            f"constraint matrix densified to dtype={arr.dtype} ndim={arr.ndim}; a "
            "sparse matrix probably reached np.asarray without going through dense_A()"
        )
    return np.asarray(arr, dtype=np.float64)


class QuadraticConstraintData(NamedTuple):
    """Quadratic row data: 0.5 x'Qx + c'x sense rhs."""

    Q: jnp.ndarray
    c: jnp.ndarray
    sense: str
    rhs: float


class QCPData(NamedTuple):
    """Standard-form QCP/QCQP data with explicit linear and quadratic rows."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_ub: jnp.ndarray  # (m_ub, n) linear inequality matrix
    b_ub: jnp.ndarray  # (m_ub,) linear inequality RHS
    A_eq: jnp.ndarray  # (m_eq, n) linear equality matrix
    b_eq: jnp.ndarray  # (m_eq,) linear equality RHS
    quadratic_constraints: tuple[QuadraticConstraintData, ...]
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def _get_variable_bounds(model: Model):
    """Extract flat lower and upper bounds from model variables.

    Returns numpy arrays to avoid JAX device-transfer overhead during
    extraction.  POUNCE's ``solve_lp_kkt``/``solve_qp_kkt`` consume numpy
    directly; only the ``custom_jvp`` differentiation wrappers convert to
    ``jnp.array``, and only for the tangent solve.
    """
    lb_parts = []
    ub_parts = []
    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
    n = sum(v.size for v in model._variables)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    lb = np.concatenate(lb_parts).astype(np.float64)
    ub = np.concatenate(ub_parts).astype(np.float64)
    return lb, ub


# ---------------------------------------------------------------------------
# Algebraic coefficient extraction (no autodiff)
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Return the starting offset of a variable in the flat x vector.

    Delegates to ``Model._flat_var_offset``, which memoizes an exclusive prefix-sum
    table and rebuilds it only when the (append-only) variable list grows, so each
    lookup is O(1). This function used to re-sum ``model._variables[: var._index]``
    from scratch — O(n_vars) per *variable reference* — which is the very quadratic
    that #654 removed everywhere else and that never got removed here.

    That omission is why extraction on ``watercontamination0202`` never returned.
    It has 106,711 variables and 107,209 constraints; the walk resolves an offset
    per variable reference per row, at a measured mean ``var._index`` of 59,909.
    Measured on that instance, sampling rows with a stride across the whole range
    (sampling only the FIRST rows hides it — they reference the lowest indices,
    where the scan is cheapest):

        rescan (before)      4.108 ms/row  ->  440 s over 107,209 rows
        memoized (after)     0.012 ms/row  ->    1.3 s
                                                340x

    Pure speedup: ``_variables`` only ever grows and a Variable's ``_index`` /
    ``size`` are immutable after construction, so the returned offset is identical
    to the rescan's (see the docstring on ``Model._flat_var_offset``).
    """
    return model._flat_var_offset(var)


class _NotLinearError(Exception):
    """Raised when an expression is not linear."""


class _NotQuadraticError(Exception):
    """Raised when an expression is not quadratic (at most degree 2)."""


def _extract_linear_coefficients(expr, model: Model, n: int):
    """Walk an expression tree to extract linear coefficients and constant.

    Returns (coefficients, constant) where:
      - coefficients is a numpy array of shape (n,) with coefficient for each variable slot
      - constant is a float scalar

    Dense wrapper over :func:`_extract_linear_coefficients_sparse`; bit-identical to
    it, because the dict accumulates exactly the same additions in the same order
    starting from the same 0.0. Callers that assemble many rows should use the sparse
    form directly and never materialise the full-width vector (#863).

    Raises _NotLinearError if the expression is not linear.
    """
    terms, const = _extract_linear_coefficients_sparse(expr, model, n)
    c = np.zeros(n, dtype=np.float64)
    for _i, _v in terms.items():
        c[_i] = _v
    return c, const


def _extract_linear_coefficients_sparse(expr, model: Model, n: int):
    """As :func:`_extract_linear_coefficients`, but keeping the row SPARSE.

    Returns ``(terms, constant)`` where ``terms`` is ``{flat_index: coefficient}``
    in first-touch order. This is what lets ``_extract_constraints_algebraic``
    assemble a (107,209 x 106,711) matrix without a 91.5 GB dense intermediate
    (#863) — the dense full-width row per constraint was never needed, only its
    nonzeros are.

    Raises _NotLinearError if the expression is not linear.
    """
    terms: dict[int, float] = {}
    const = 0.0

    def _add(i: int, v: float) -> None:
        # The dense predecessor got this bound check for free from numpy (and, for a
        # negative index, silently wrote the WRONG slot via numpy wraparound).
        # _NotLinearError rather than IndexError so the dispatcher falls through to
        # the next extractor exactly as it did on the IndexError before.
        if not 0 <= i < n:
            raise _NotLinearError(f"variable slot {i} outside the model's {n} flat slots")
        terms[i] = terms.get(i, 0.0) + v

    def _walk(node, scale=1.0, allow_array=False):
        # ``allow_array`` is True only inside a ``sum(...)`` reduction, where a
        # size>1 (sub)expression contributes a single scalar row (its element sum
        # with a *uniform* scale). Outside a sum, encountering a size>1 array node
        # in scalar position means the whole body is vector-valued: the algebraic
        # extractor would collapse it to one summed row (C-29), certifying an
        # infeasible point. Refuse instead so extract_lp_data() routes the body to
        # the per-component autodiff extractor (one LP row per element).
        nonlocal const

        if isinstance(node, Constant):
            val = node.value
            if val.ndim == 0 or val.size == 1:
                const += scale * float(val.reshape(()))
            else:
                raise _NotLinearError("Array constant in unexpected position")
            return

        if isinstance(node, Variable):
            offset = _compute_var_offset(node, model)
            if node.size == 1:
                _add(offset, scale)
            elif allow_array:
                # Inside sum(): sum(scale * x) = scale * Σ x_j (uniform scale).
                for j in range(node.size):
                    _add(offset + j, scale)
            else:
                raise _NotLinearError(
                    "Array variable in scalar position (vector-valued body); "
                    "routing to the per-component extractor"
                )
            return

        if isinstance(node, IndexExpression):
            if isinstance(node.base, Variable):
                var = node.base
                offset = _compute_var_offset(var, model)
                idx = node.index
                if isinstance(idx, (int, np.integer)):
                    _add(offset + int(idx), scale)
                elif (
                    isinstance(idx, tuple)
                    and len(idx) == 1
                    and isinstance(idx[0], (int, np.integer))
                ):
                    _add(offset + int(idx[0]), scale)
                else:
                    # Multi-dimensional index: flatten
                    try:
                        flat_idx = np.ravel_multi_index(
                            idx if isinstance(idx, tuple) else (idx,), var.shape
                        )
                    except (TypeError, ValueError):
                        # Sliced/partial subscript (vectorized term): this scalar
                        # extractor cannot express it; classify as not-linear.
                        raise _NotLinearError(f"non-scalar index {idx!r} on {var.name}") from None
                    _add(offset + int(flat_idx), scale)
                return
            raise _NotLinearError(f"IndexExpression on non-variable: {type(node.base)}")

        if isinstance(node, BinaryOp):
            if node.op == "+":
                _walk(node.left, scale, allow_array)
                _walk(node.right, scale, allow_array)
                return
            if node.op == "-":
                _walk(node.left, scale, allow_array)
                _walk(node.right, -scale, allow_array)
                return
            if node.op == "*":
                # One side must be a scalar constant for linearity. A size>1 array
                # constant here raises _NotLinearError from _eval_const (the scale
                # would differ per element), which routes to the per-component
                # extractor rather than collapsing.
                if _is_const_expr(node.left):
                    cval = _eval_const(node.left)
                    _walk(node.right, scale * cval, allow_array)
                    return
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale * cval, allow_array)
                    return
                raise _NotLinearError("Product of two variable expressions")
            if node.op == "/":
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale / cval, allow_array)
                    return
                raise _NotLinearError("Division by variable expression")
            raise _NotLinearError(f"Non-linear operator: {node.op}")

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                _walk(node.operand, -scale, allow_array)
                return
            raise _NotLinearError(f"Non-linear unary op: {node.op}")

        if isinstance(node, SumOverExpression):
            for term in node.terms:
                _walk(term, scale, allow_array)
            return

        if isinstance(node, SumExpression):
            # sum(expr) reduces expr to a scalar: element-collapse is legitimate
            # here (uniform scale), so allow array nodes beneath this point.
            _walk(node.operand, scale, allow_array=True)
            return

        if isinstance(node, MatMulExpression):
            # Handle Constant @ Variable or Variable @ Constant
            if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                mat = node.left.value
                var = node.right
                offset = _compute_var_offset(var, model)
                # mat @ var => result is mat @ x[offset:offset+size]
                # For 1-D mat (dot product), coefficients are mat elements
                if mat.ndim == 1:
                    for j in range(var.size):
                        _add(offset + j, scale * float(mat[j]))
                elif mat.ndim == 2:
                    # Returns vector; this should be used inside a sum
                    raise _NotLinearError("MatMul returning vector in scalar context")
                return
            if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                mat = node.right.value
                var = node.left
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        _add(offset + j, scale * float(mat[j]))
                    return
                raise _NotLinearError("MatMul returning vector in scalar context")
            raise _NotLinearError("MatMul between non-trivial expressions")

        raise _NotLinearError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return terms, const


def _materialise_Q(terms: dict[tuple[int, int], float], n: int) -> np.ndarray:
    """Assemble a Hessian from ``{(row, col): value}`` accumulated entries (#863).

    Dense while ``(n, n)`` float64 fits ``_QP_DENSE_Q_MAX_BYTES`` — bit-identical to
    the ``np.zeros((n, n))`` this replaced, because the dict performs the same
    ``+=`` additions in the same order starting from the same 0.0 — and scipy CSR
    beyond it. ``dense_Q()`` re-densifies for consumers.

    The dict is what makes the sparse arm reachable at all. ``np.zeros((n, n))`` is
    91 GB on ``watercontamination0202`` (106,711 variables); macOS *allows* that
    allocation because zero pages are mapped lazily, so it does not raise — it just
    makes the first full read of the array catastrophic. Measured on that instance,
    a single ``Q @ x`` against the lazily-allocated dense Q (holding 4,017 nonzeros)
    took **16.0 s**. Accumulating entries instead keeps peak memory at O(nnz), so
    the dense matrix never has to exist even transiently.

    Mirrors :func:`_materialise_A`, which did the same for the constraint matrix.
    """
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if (n * n * 8) <= _QP_DENSE_Q_MAX_BYTES:
        Q = np.zeros((n, n), dtype=np.float64)
        for (_i, _j), _v in terms.items():
            Q[_i, _j] = _v
        return Q
    import scipy.sparse as _sp

    if not terms:
        # csr_matrix((data, (row, col))) needs non-empty index arrays to infer dtype.
        return cast(np.ndarray, _sp.csr_matrix((n, n), dtype=np.float64))
    r = np.fromiter((k[0] for k in terms), dtype=np.intp, count=len(terms))
    c = np.fromiter((k[1] for k in terms), dtype=np.intp, count=len(terms))
    v = np.fromiter(terms.values(), dtype=np.float64, count=len(terms))
    # Annotated ndarray because that is what every consumer sees after dense_Q();
    # the sparse arm is deliberately outside the annotation, exactly as the repr
    # extractor's producer is (c525f519).
    return cast(np.ndarray, _sp.csr_matrix((v, (r, c)), shape=(n, n)))


def _quadratic_terms_nonempty(terms: dict[tuple[int, int], float], tol: float = 1e-12) -> bool:
    """``_quadratic_row_has_terms`` on the pre-materialisation accumulator.

    Identical predicate (entries absent from the dict are exactly 0.0 in both arms),
    but it never needs the matrix, so a QCP row can be classified linear-or-quadratic
    without materialising anything (#863).
    """
    return any(abs(_v) > tol for _v in terms.values())


def _extract_quadratic_coefficients(expr, model: Model, n: int):
    """Walk expression tree to extract quadratic and linear coefficients.

    Returns (Q, c, constant) where:
      - Q is the Hessian (f = 0.5 x'Qx + c'x + const): a dense (n, n) numpy array
        while that fits ``_QP_DENSE_Q_MAX_BYTES``, scipy CSR beyond it (#863) --
        consumers densify through :func:`dense_Q`
      - c is (n,) numpy array of linear coefficients
      - constant is a float scalar

    Callers that only need the *predicate* "does this have quadratic terms" or the
    nonzero entries should use :func:`_extract_quadratic_terms` and avoid
    materialising anything.

    Raises _NotQuadraticError if the expression has degree > 2.
    """
    terms, c, const = _extract_quadratic_terms(expr, model, n)
    return _materialise_Q(terms, n), c, const


def _extract_quadratic_terms(expr, model: Model, n: int):
    """As :func:`_extract_quadratic_coefficients`, but keeping the Hessian SPARSE.

    Returns ``(terms, c, constant)`` with ``terms`` a ``{(row, col): value}`` dict in
    first-touch order. The full-width ``(n, n)`` Hessian was never needed by the
    walk, only its nonzeros; see :func:`_materialise_Q` for why it must not be
    allocated on a wide model.
    """
    q_terms: dict[tuple[int, int], float] = {}
    c = np.zeros(n, dtype=np.float64)
    const = 0.0

    def _qadd(i: int, j: int, v: float) -> None:
        # The dense predecessor got this bound check for free from numpy (and, for a
        # negative index, silently wrote the WRONG cell via numpy wraparound).
        # _NotQuadraticError rather than IndexError so the dispatcher falls through
        # to the next extractor exactly as it did on the IndexError before.
        if not (0 <= i < n and 0 <= j < n):
            raise _NotQuadraticError(f"Hessian cell ({i}, {j}) outside the model's {n} flat slots")
        q_terms[(i, j)] = q_terms.get((i, j), 0.0) + v

    def _get_var_index(node):
        """Get the flat variable index for a variable-like node, or None."""
        if isinstance(node, Variable):
            if node.size != 1:
                return None
            return _compute_var_offset(node, model)
        if isinstance(node, IndexExpression):
            # #941: the bare-int fast path was `offset + int(idx)`, wrong for a
            # negative index. (The `ravel_multi_index` fallback below it already
            # refused negatives, by raising — so only the fast path was unsound.)
            return resolve_scalar_slot(node, model)
        return None

    def _walk(node, scale=1.0, allow_array=False):
        # See _extract_linear_coefficients._walk: ``allow_array`` is True only
        # inside a sum() reduction, where a size>1 array variable legitimately
        # collapses to a single scalar term. Outside a sum, an array variable in
        # scalar position means a vector-valued body that must NOT be collapsed to
        # one row (C-29) — refuse so the caller routes to the autodiff extractor.
        nonlocal const

        if isinstance(node, Constant):
            val = node.value
            if val.ndim == 0 or val.size == 1:
                const += scale * float(val.reshape(()))
            else:
                raise _NotQuadraticError("Array constant in unexpected position")
            return

        if isinstance(node, (Variable, IndexExpression)):
            idx = _get_var_index(node)
            if idx is not None:
                c[idx] += scale
                return
            if isinstance(node, Variable) and node.size > 1:
                if not allow_array:
                    raise _NotQuadraticError(
                        "Array variable in scalar position (vector-valued body); "
                        "routing to the per-component extractor"
                    )
                offset = _compute_var_offset(node, model)
                for j in range(node.size):
                    c[offset + j] += scale
                return
            raise _NotQuadraticError(f"Cannot extract index from {node}")

        if isinstance(node, BinaryOp):
            if node.op == "+":
                _walk(node.left, scale, allow_array)
                _walk(node.right, scale, allow_array)
                return
            if node.op == "-":
                _walk(node.left, scale, allow_array)
                _walk(node.right, -scale, allow_array)
                return
            if node.op == "*":
                # Check: const * expr, expr * const, or var * var
                if _is_const_expr(node.left):
                    cval = _eval_const(node.left)
                    _walk(node.right, scale * cval, allow_array)
                    return
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale * cval, allow_array)
                    return
                # var * var => quadratic term
                # Q is the Hessian: f = 0.5 x'Qx, so d²(c*xi*xj)/dxi dxj = c,
                # but d²(c*xi²)/dxi² = 2c. We store the Hessian directly.
                idx_l = _get_var_index(node.left)
                idx_r = _get_var_index(node.right)
                if idx_l is not None and idx_r is not None:
                    if idx_l == idx_r:
                        _qadd(idx_l, idx_r, 2.0 * scale)
                    else:
                        _qadd(idx_l, idx_r, scale)
                        _qadd(idx_r, idx_l, scale)
                    return
                # Handle (const * var) * var or var * (const * var):
                # e.g., (Q[i,j] * x[i]) * x[j] from left-to-right evaluation
                cv_l = _try_extract_const_var(node.left, model)
                if cv_l is not None and idx_r is not None:
                    cval, idx_l2 = cv_l
                    if idx_l2 == idx_r:
                        _qadd(idx_l2, idx_r, 2.0 * scale * cval)
                    else:
                        _qadd(idx_l2, idx_r, scale * cval)
                        _qadd(idx_r, idx_l2, scale * cval)
                    return
                cv_r = _try_extract_const_var(node.right, model)
                if cv_r is not None and idx_l is not None:
                    cval, idx_r2 = cv_r
                    if idx_l == idx_r2:
                        _qadd(idx_l, idx_r2, 2.0 * scale * cval)
                    else:
                        _qadd(idx_l, idx_r2, scale * cval)
                        _qadd(idx_r2, idx_l, scale * cval)
                    return
                raise _NotQuadraticError("Product of non-simple variable expressions")
            if node.op == "/":
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale / cval, allow_array)
                    return
                raise _NotQuadraticError("Division by variable expression")
            if node.op == "**":
                # x**2 => quadratic
                if _is_const_expr(node.right):
                    pval = _eval_const(node.right)
                    if abs(pval - 2.0) < 1e-12:
                        idx = _get_var_index(node.left)
                        if idx is not None:
                            _qadd(idx, idx, 2.0 * scale)  # x^2 = 0.5 * 2 * x^2
                            return
                    if abs(pval - 1.0) < 1e-12:
                        _walk(node.left, scale, allow_array)
                        return
                    if abs(pval) < 1e-12:
                        const += scale
                        return
                raise _NotQuadraticError(f"Power with exponent {node.right}")
            raise _NotQuadraticError(f"Unknown binary op: {node.op}")

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                _walk(node.operand, -scale, allow_array)
                return
            raise _NotQuadraticError(f"Non-linear unary op: {node.op}")

        if isinstance(node, SumOverExpression):
            for term in node.terms:
                _walk(term, scale, allow_array)
            return

        if isinstance(node, SumExpression):
            # sum(expr) reduces to a scalar: array collapse legitimate below here.
            _walk(node.operand, scale, allow_array=True)
            return

        if isinstance(node, MatMulExpression):
            # Handle Constant @ Variable for linear parts of QP constraints
            if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                mat = node.left.value
                var = node.right
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                    return
                raise _NotQuadraticError("MatMul returning vector")
            if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                mat = node.right.value
                var = node.left
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                    return
                raise _NotQuadraticError("MatMul returning vector")
            raise _NotQuadraticError("MatMul between non-trivial expressions")

        raise _NotQuadraticError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return q_terms, c, const


def _try_extract_const_var(expr, model: Model):
    """Try to decompose expr as (constant * variable).

    Returns (constant_value, flat_var_index) if expr is of the form
    Constant * Variable/IndexExpr or Variable/IndexExpr * Constant,
    or just a bare Variable/IndexExpr (constant = 1.0).

    Returns None if the expression is not of this form.
    """
    # Bare variable => coefficient 1.0
    if isinstance(expr, (Variable, IndexExpression)):
        if isinstance(expr, Variable) and expr.size != 1:
            return None
        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            offset = _compute_var_offset(expr.base, model)
            idx = expr.index
            if isinstance(idx, (int, np.integer)):
                return (1.0, offset + int(idx))
            if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], (int, np.integer)):
                return (1.0, offset + int(idx[0]))
            try:
                flat_idx = np.ravel_multi_index(
                    idx if isinstance(idx, tuple) else (idx,), expr.base.shape
                )
            except (TypeError, ValueError):
                return None  # sliced/partial subscript: not a scalar reference
            return (1.0, offset + int(flat_idx))
        if isinstance(expr, Variable):
            return (1.0, _compute_var_offset(expr, model))
        return None

    # const * var or var * const
    if isinstance(expr, BinaryOp) and expr.op == "*":
        if _is_const_expr(expr.left):
            cval = _eval_const(expr.left)
            inner = _try_extract_const_var(expr.right, model)
            if inner is not None:
                return (cval * inner[0], inner[1])
        if _is_const_expr(expr.right):
            cval = _eval_const(expr.right)
            inner = _try_extract_const_var(expr.left, model)
            if inner is not None:
                return (cval * inner[0], inner[1])

    # neg(var) => -1.0 * var
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _try_extract_const_var(expr.operand, model)
        if inner is not None:
            return (-inner[0], inner[1])

    return None


def _is_const_expr(expr) -> bool:
    """Check if an expression is a pure constant (no variables)."""
    if isinstance(expr, Constant):
        return True
    if isinstance(expr, (Variable, IndexExpression)):
        return False
    if isinstance(expr, BinaryOp):
        return _is_const_expr(expr.left) and _is_const_expr(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_const_expr(expr.operand)
    if isinstance(expr, SumOverExpression):
        return all(_is_const_expr(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _is_const_expr(expr.operand)
    return False


def _eval_const(expr) -> float:  # type: ignore[return-value]
    """Evaluate a constant expression to a float scalar.

    Raises ``_NotLinearError`` (NOT ``ValueError``) on a non-scalar array
    constant. C-30: a raw ``ValueError`` from ``float(v.item())`` on a size>1
    array (e.g. ``sum(np.array([1,1]) * x)``) used to abort the algebraic walk
    and mis-route to a fallback that dropped the objective sense. Refusing with
    ``_NotLinearError`` routes such bodies to the row- and sense-correct autodiff
    extractor instead.
    """
    if isinstance(expr, Constant):
        v = expr.value
        if v.ndim == 0 or v.size == 1:
            return float(v.reshape(()))
        raise _NotLinearError(
            "Non-scalar array constant cannot be evaluated as a scalar coefficient; "
            "routing to the per-component extractor"
        )
    if isinstance(expr, BinaryOp):
        lv = _eval_const(expr.left)
        r = _eval_const(expr.right)
        if expr.op == "+":
            return lv + r
        if expr.op == "-":
            return lv - r
        if expr.op == "*":
            return lv * r
        if expr.op == "/":
            return lv / r
        if expr.op == "**":
            return float(lv**r)
        raise ValueError(f"Unknown op in const eval: {expr.op}")
    if isinstance(expr, UnaryOp):
        uv = _eval_const(expr.operand)
        if expr.op == "neg":
            return float(-uv)
        if expr.op == "abs":
            return float(abs(uv))
        raise ValueError(f"Unknown unary op in const eval: {expr.op}")
    if isinstance(expr, SumOverExpression):
        return sum(_eval_const(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _eval_const(expr.operand)
    raise ValueError(f"Not a constant expression: {type(expr).__name__}")


def _extract_constraints_algebraic(model: Model, n_orig: int):
    """Extract linear constraint data algebraically (shared by LP and QP paths).

    Returns (A_eq, b_eq, x_l, x_u, n_slack) where slacks are appended for
    inequality constraints. ``A_eq`` is dense while it fits ``_DENSE_A_MAX_BYTES``
    and scipy CSR beyond it (#863) — consumers densify via ``dense_A()``.

    Raises _NotLinearError if any constraint is not linear.
    """
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    eq_terms: list[dict[int, float]] = []
    eq_rhs: list[float] = []
    ineq_terms: list[dict[int, float]] = []
    ineq_senses: list[str] = []
    ineq_rhs: list[float] = []

    for con in constraints:
        # Sparse walk: the dense full-width row per constraint was never needed,
        # only its nonzeros (#863).
        terms, const = _extract_linear_coefficients_sparse(con.body, model, n_orig)
        if con.sense == "==":
            eq_terms.append(terms)
            eq_rhs.append(-const)
        elif con.sense == "<=":
            ineq_terms.append(terms)
            ineq_senses.append("le")
            ineq_rhs.append(-const)
        elif con.sense == ">=":
            ineq_terms.append(terms)
            ineq_senses.append("ge")
            ineq_rhs.append(-const)

    n_eq = len(eq_terms)
    n_ineq = len(ineq_terms)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    # COO triples, not np.stack() of dense (n_total,) rows: that stack is
    # 107,209 x 106,711 float64 = 91.5 GB on watercontamination0202, and it needs
    # every row resident simultaneously (#863).
    coo_rows: list[int] = []
    coo_cols: list[int] = []
    coo_vals: list[float] = []
    b_vals: list[float] = []

    for i in range(n_eq):
        _append_row_coo(coo_rows, coo_cols, coo_vals, i, eq_terms[i])
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        r = n_eq + i
        _append_row_coo(coo_rows, coo_cols, coo_vals, r, ineq_terms[i])
        # body <= 0 becomes body + s = 0; body >= 0 becomes body - s = 0; s >= 0.
        coo_rows.append(r)
        coo_cols.append(n_orig + i)
        coo_vals.append(1.0 if ineq_senses[i] == "le" else -1.0)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    A_eq = _materialise_A(coo_rows, coo_cols, coo_vals, m_total, n_total)
    b_eq = np.array(b_vals, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, 1e20, dtype=np.float64)])

    return A_eq, b_eq, x_l, x_u, n_slack


def _quadratic_row_has_terms(Q: np.ndarray, tol: float = 1e-12) -> bool:
    """True when ``Q`` holds a nonzero entry above ``tol``.

    Sparse-aware (#875): ``np.abs`` on a scipy sparse matrix returns a sparse matrix,
    and ``np.any`` on one does not mean what it does on an ndarray — the stored
    values are the only candidates, so test those directly. Explicit zeros can be
    stored, hence the ``> tol`` test rather than ``nnz``.
    """
    if _sp_issparse(Q):
        return bool(np.any(np.abs(Q.data) > tol))
    return bool(np.any(np.abs(Q) > tol))


def _empty_matrix(n_cols: int) -> np.ndarray:
    return np.zeros((0, n_cols), dtype=np.float64)


def _materialise_A(
    rows: list[int], cols: list[int], vals: list[float], m: int, n: int
) -> np.ndarray:
    """Assemble a constraint matrix from COO triples (#863).

    Dense while ``(m, n)`` float64 fits ``_DENSE_A_MAX_BYTES`` — bit-identical to the
    ``np.stack`` of dense full-width rows this replaced — and scipy CSR beyond it,
    because a model with both many rows and many columns cannot hold ``(m, n)``
    floats at all: ``watercontamination0202`` is 107,209 x 106,711, which is 91.5 GB.
    ``dense_A()`` re-densifies for consumers.

    The COO form is what makes the sparse arm reachable at all. ``np.stack`` needs
    every dense row resident simultaneously; contributing rows as triples keeps peak
    memory at O(nnz), so the dense matrix never has to exist even transiently.

    Entries absent from the triples are 0.0 in both arms, so dropping explicitly
    zero coefficients (as ``_append_dense_row_coo`` does) cannot change the
    densified result.
    """
    if m == 0:
        return _empty_matrix(n)
    r = np.asarray(rows, dtype=np.intp)
    c = np.asarray(cols, dtype=np.intp)
    v = np.asarray(vals, dtype=np.float64)
    if (m * n * 8) <= _DENSE_A_MAX_BYTES:
        A = np.zeros((m, n), dtype=np.float64)
        A[r, c] = v
        return A
    import scipy.sparse as _sp

    # Annotated ndarray because that is what every consumer sees after dense_A();
    # the sparse arm is deliberately outside the annotation, exactly as dense_Q's
    # producer is (c525f519).
    return cast(np.ndarray, _sp.csr_matrix((v, (r, c)), shape=(m, n)))


def _append_row_coo(
    rows: list[int], cols: list[int], vals: list[float], r: int, terms: dict[int, float]
) -> None:
    """Append row ``r``'s entries from a ``{column: coefficient}`` mapping."""
    for _j, _v in terms.items():
        rows.append(r)
        cols.append(_j)
        vals.append(_v)


def _append_dense_row_coo(
    rows: list[int], cols: list[int], vals: list[float], r: int, vec: np.ndarray
) -> None:
    """Append only the nonzeros of the dense coefficient row ``vec`` as row ``r``.

    For the extractors whose row source is already a dense vector (the repr probe
    paths, the QCP algebraic walk). Converting each row as it is produced and
    dropping it holds peak memory at O(nnz) rather than O(m*n).

    ``np.nonzero`` retains NaN/inf (they compare unequal to 0), so the finiteness
    guard in ``_extract_lp_data_from_repr`` still sees them.
    """
    (nz,) = np.nonzero(vec)
    if nz.size == 0:
        return
    rows.extend([r] * int(nz.size))
    cols.extend(nz.tolist())
    vals.extend(np.asarray(vec, dtype=np.float64)[nz].tolist())


def _all_finite(arr) -> bool:
    """``np.isfinite(arr).all()`` for a dense array or a scipy sparse matrix.

    Unstored sparse entries are exactly 0.0, which is finite, so checking ``.data``
    is equivalent to checking the densified matrix — without densifying it.
    """
    if _sp_issparse(arr):
        return bool(np.isfinite(arr.data).all())
    a = np.asarray(arr)
    return bool(a.size == 0 or np.isfinite(a).all())


def _extract_qcp_constraints_algebraic(
    model: Model,
    n_orig: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[QuadraticConstraintData, ...]]:
    """Extract linear and quadratic rows without introducing slack variables."""

    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # COO accumulators rather than lists of dense rows (#863): a row is reduced to
    # its nonzeros as soon as it is produced, so an (m, n_orig) dense stack is never
    # built. The per-QUADRATIC-row Hessian is accumulated sparsely for the same
    # reason and materialised once, dense or CSR per ``_materialise_Q``'s budget —
    # a LINEAR row never materialises one at all.
    ub_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    ub_rhs: list[float] = []
    eq_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    eq_rhs: list[float] = []
    q_rows: list[QuadraticConstraintData] = []

    for con in constraints:
        q_terms, c_vec, const = _extract_quadratic_terms(con.body, model, n_orig)
        rhs = float(con.rhs) - float(const)
        if _quadratic_terms_nonempty(q_terms):
            Q = _materialise_Q(q_terms, n_orig)
            q_rows.append(
                QuadraticConstraintData(
                    # Preserve sparsity: np.asarray() on a sparse matrix silently
                    # yields a 0-d object array rather than raising (#863).
                    # Consumers densify via dense_Q().
                    Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
                    c=np.asarray(c_vec),  # type: ignore[arg-type]
                    sense=con.sense,
                    rhs=rhs,
                )
            )
            continue

        if con.sense == "==":
            _append_dense_row_coo(*eq_coo, len(eq_rhs), c_vec)
            eq_rhs.append(rhs)
        elif con.sense == "<=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), c_vec)
            ub_rhs.append(rhs)
        elif con.sense == ">=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), -c_vec)
            ub_rhs.append(-rhs)
        else:
            raise _NotQuadraticError(f"Unknown constraint sense: {con.sense}")

    A_ub = _materialise_A(*ub_coo, len(ub_rhs), n_orig)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)
    A_eq = _materialise_A(*eq_coo, len(eq_rhs), n_orig)
    b_eq = np.asarray(eq_rhs, dtype=np.float64)
    return A_ub, b_ub, A_eq, b_eq, tuple(q_rows)


def extract_lp_data_algebraic(model: Model) -> LPData:
    """Extract LP standard form by walking the expression DAG algebraically.

    Much faster than extract_lp_data() because it avoids JAX tracing/autodiff.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotLinearError if the model is not linear.
    """

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    c, obj_const = _extract_linear_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)
    c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data_algebraic(model: Model) -> QPData:
    """Extract QP standard form by walking the expression DAG algebraically.

    Much faster than extract_qp_data() because it avoids jax.hessian tracing.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotQuadraticError if the objective is not quadratic.
    """

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    Q, c_vec, obj_const = _extract_quadratic_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)

    if n_slack > 0:
        n_total = n_orig + n_slack
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            # Pad sparsely: densifying here would defeat the whole point (#863).
            Q_full = _sp.block_diag((Q, _sp.csr_matrix((n_slack, n_slack))), format="csr")
        else:
            Q_full = np.zeros((n_total, n_total), dtype=np.float64)
            Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q_full = -Q_full
        c_full = -c_full
        obj_const = -obj_const

    return QPData(
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_Q().
        Q=Q_full if _sp_issparse(Q_full) else np.asarray(Q_full),  # type: ignore[arg-type]
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qcp_data_algebraic(model: Model) -> QCPData:
    """Extract QCP/QCQP data by walking the expression DAG algebraically."""

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    Q, c_vec, obj_const = _extract_quadratic_coefficients(obj_expr, model, n_orig)
    A_ub, b_ub, A_eq, b_eq, q_rows = _extract_qcp_constraints_algebraic(model, n_orig)
    x_l, x_u = _get_variable_bounds(model)

    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q = -Q
        c_vec = -c_vec
        obj_const = -obj_const

    return QCPData(
        # ``Q`` may now be sparse too (#863) — same np.asarray hazard, same fix.
        Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
        c=np.asarray(c_vec),  # type: ignore[arg-type]
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_A().
        A_ub=A_ub if _sp_issparse(A_ub) else np.asarray(A_ub),  # type: ignore[arg-type]
        b_ub=np.asarray(b_ub),  # type: ignore[arg-type]
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        quadratic_constraints=q_rows,
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=obj_const,
    )


def _extract_lp_data_from_repr(model: Model) -> LPData:
    """Extract LP data by evaluating the Rust ModelRepr at unit vectors.

    For linear functions, c_j = f(e_j) - f(0) and A_ij = g_i(e_j) - g_i(0).
    This works for fast-API models where Python expression trees don't exist.
    """

    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars
    n_con = repr_.n_constraints

    x_zero = np.zeros(n_orig, dtype=np.float64)
    obj_at_zero = repr_.evaluate_objective(x_zero)

    # Extract objective coefficients
    c = np.zeros(n_orig, dtype=np.float64)
    for j in range(n_orig):
        ej = np.zeros(n_orig, dtype=np.float64)
        ej[j] = 1.0
        c[j] = repr_.evaluate_objective(ej) - obj_at_zero

    # Extract constraint data. Rows are reduced to their nonzeros as soon as they
    # are probed (#863): retaining n_con dense (n_orig,) vectors and np.stack()ing
    # them is 91.5 GB on watercontamination0202 and needs every row resident at once.
    eq_terms: list[dict[int, float]] = []
    eq_rhs: list[float] = []
    ineq_terms: list[dict[int, float]] = []
    ineq_senses: list[str] = []
    ineq_rhs: list[float] = []

    def _row_terms(vec) -> dict[int, float]:
        (nz,) = np.nonzero(vec)
        return {int(j): float(vec[j]) for j in nz}

    for i in range(n_con):
        sense = repr_.constraint_sense(i)
        rhs_val = repr_.constraint_rhs(i)
        g_at_zero = repr_.evaluate_constraint(i, x_zero)

        a_row = np.zeros(n_orig, dtype=np.float64)
        for j in range(n_orig):
            ej = np.zeros(n_orig, dtype=np.float64)
            ej[j] = 1.0
            a_row[j] = repr_.evaluate_constraint(i, ej) - g_at_zero

        if sense == "==":
            eq_terms.append(_row_terms(a_row))
            eq_rhs.append(rhs_val - g_at_zero)
        elif sense == "<=":
            ineq_terms.append(_row_terms(a_row))
            ineq_senses.append("le")
            ineq_rhs.append(rhs_val - g_at_zero)
        elif sense == ">=":
            ineq_terms.append(_row_terms(a_row))
            ineq_senses.append("ge")
            ineq_rhs.append(rhs_val - g_at_zero)

    n_eq = len(eq_terms)
    n_ineq = len(ineq_terms)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    coo_rows: list[int] = []
    coo_cols: list[int] = []
    coo_vals: list[float] = []
    b_vals: list[float] = []

    for i in range(n_eq):
        _append_row_coo(coo_rows, coo_cols, coo_vals, i, eq_terms[i])
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        r = n_eq + i
        _append_row_coo(coo_rows, coo_cols, coo_vals, r, ineq_terms[i])
        # body <= 0 becomes body + s = 0; body >= 0 becomes body - s = 0; s >= 0.
        coo_rows.append(r)
        coo_cols.append(n_orig + i)
        coo_vals.append(1.0 if ineq_senses[i] == "le" else -1.0)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    A_eq = _materialise_A(coo_rows, coo_cols, coo_vals, m_total, n_total)
    b_eq = np.array(b_vals, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, np.inf, dtype=np.float64)])

    obj_sense = repr_.objective_sense
    if obj_sense == "maximize":
        c_full = -c_full
        obj_at_zero = -obj_at_zero

    # This evaluator reduces each constraint to a single scalar row by probing
    # the Rust repr at unit vectors. Vector-/matrix-valued constraints (DAE
    # collocation residuals, `Variable @ Constant` MOL stencils) cannot be
    # represented that way — `evaluate_constraint` returns NaN for them — so a
    # non-finite coefficient here means the repr path silently mis-extracted the
    # model. Decline instead of returning corrupt data: `extract_lp_data` then
    # falls through to the autodiff path, which expands such constraints into one
    # row per component. (A NaN reaching the LP solver otherwise crashes/hangs
    # HiGHS — issue surfaced via test_mol_collocation_solves.)
    for _name, _arr in (("c", c_full), ("A_eq", A_eq), ("b_eq", b_eq)):
        if not _all_finite(_arr):
            raise _NotLinearError(
                f"repr-based LP extraction produced non-finite {_name}; the model "
                "has vector-valued constraints that are not scalar-representable"
            )

    return LPData(
        c=np.asarray(c_full),  # type: ignore[arg-type]
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_A().
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=obj_at_zero,
    )


def _extract_qp_data_from_repr(model: Model) -> QPData:
    """Extract QP data by evaluating the Rust ModelRepr numerically.

    For the quadratic objective 0.5 x'Qx + c'x + d:
      - d = f(0)
      - c_j = f(e_j) - d - 0.5*Q[j,j]   but Q[j,j] = f(e_j) + f(-e_j) - 2*d
      - Q[i,j] = f(e_i+e_j) - f(e_i) - f(e_j) + d  (for i != j)

    Constraints are extracted as in the LP case.
    """

    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars
    x_zero = np.zeros(n_orig, dtype=np.float64)
    d = repr_.evaluate_objective(x_zero)

    # Evaluate at all unit vectors
    f_ej = np.zeros(n_orig, dtype=np.float64)
    f_neg_ej = np.zeros(n_orig, dtype=np.float64)
    for j in range(n_orig):
        ej = np.zeros(n_orig, dtype=np.float64)
        ej[j] = 1.0
        f_ej[j] = repr_.evaluate_objective(ej)
        ej[j] = -1.0
        f_neg_ej[j] = repr_.evaluate_objective(ej)

    # Q diagonal: Q[j,j] = f(e_j) + f(-e_j) - 2*d  (kept 1-D; see the note below on
    # why the dense (n, n) is not materialised until the very end)
    diag = f_ej + f_neg_ej - 2 * d

    # Q off-diagonal: Q[i,j] = f(e_i+e_j) - f(e_i) - f(e_j) + d
    #
    # Restricted to the objective's SUPPORT (#863). A variable absent from the
    # objective has f(e_j) == f(-e_j) == d and a zero diagonal, so every product
    # involving it is identically zero and probing that pair is pure waste. The
    # support is free -- the O(n) diagonal probes above already identify it.
    #
    # ``watercontamination0202`` is 106,711 variables whose objective touches 101;
    # the unrestricted sweep issues 5.69e9 probes to discover ~1e4 possibly-nonzero
    # entries, and a dense (n, n) Q there is 91 GB.
    support = [j for j in range(n_orig) if f_ej[j] != d or f_neg_ej[j] != d or diag[j] != 0.0]
    off_i: list[int] = []
    off_j: list[int] = []
    off_v: list[float] = []
    for _si, i in enumerate(support):
        for j in support[_si + 1 :]:
            eij = np.zeros(n_orig, dtype=np.float64)
            eij[i] = 1.0
            eij[j] = 1.0
            qij = repr_.evaluate_objective(eij) - f_ej[i] - f_ej[j] + d
            if qij != 0.0:
                off_i.append(i)
                off_j.append(j)
                off_v.append(float(qij))

    # Materialise Q. Dense while it comfortably fits (bit-identical to the previous
    # behaviour); sparse beyond that, because a wide model with a narrow objective
    # cannot hold (n, n) floats at all. ``dense_Q()`` re-densifies for consumers.
    _dense_ok = (n_orig * n_orig * 8) <= _QP_DENSE_Q_MAX_BYTES
    if _dense_ok:
        Q = np.zeros((n_orig, n_orig), dtype=np.float64)
        np.fill_diagonal(Q, diag)
        for _i, _j, _v in zip(off_i, off_j, off_v):
            Q[_i, _j] = _v
            Q[_j, _i] = _v
    else:
        import scipy.sparse as _sp

        rows = list(range(n_orig)) + off_i + off_j
        cols = list(range(n_orig)) + off_j + off_i
        vals = list(diag) + off_v + off_v
        Q = _sp.csr_matrix((vals, (rows, cols)), shape=(n_orig, n_orig))

    # Linear coefficients: c_j = f(e_j) - d - 0.5*Q[j,j]
    c_vec = f_ej - d - 0.5 * diag

    # --- Verify the probes actually recovered the objective (#866) ---
    # Every formula above is an exact identity in real arithmetic but a difference
    # of nearly-equal floats in practice: ``diag[j] = f(e_j) + f(-e_j) - 2d``
    # cancels catastrophically once the constant term dwarfs the quadratic
    # coefficient. On ``min (x - 1e10)**2`` (d = 1e20, ulp(1e20) ~ 16384) the unit
    # probes lose the ``+1`` entirely and this returns **Q = 0** -- the objective
    # silently becomes linear. That produced a CERTIFIED optimum of -9e20 for a sum
    # of squares (issue #866): a false optimum, the worst error class
    # (CLAUDE.md §1), on the default path.
    #
    # So do not trust the probes: re-evaluate the recovered quadratic against the
    # model's own objective at a few scale-aware points and raise if it disagrees.
    # Raising hands the dispatcher on to the next extractor.
    _probe_scale = 1.0
    for _v in getattr(model, "_variables", []):
        for _b in (getattr(_v, "lb", None), getattr(_v, "ub", None)):
            if _b is None:
                continue
            _arr = np.asarray(_b, dtype=np.float64).ravel()
            _fin = _arr[np.isfinite(_arr)]
            if _fin.size:
                _probe_scale = max(_probe_scale, min(float(np.max(np.abs(_fin))), 1e12))
    _rng = np.random.default_rng(0)
    for _k in range(3):
        _pt = (_rng.random(n_orig) * 2.0 - 1.0) * _probe_scale
        _recovered = 0.5 * float(_pt @ (Q @ _pt)) + float(c_vec @ _pt) + d
        # Reuse the SAME evaluator the probes used, so the cross-check is always
        # available. (An earlier draft reached for ``model._nl_repr``, absent for
        # API-built models, and the check silently skipped itself -- the very
        # failure mode it exists to catch.)
        _truth = float(repr_.evaluate_objective(_pt))
        if not np.isfinite(_recovered) or not np.isfinite(_truth):
            continue
        if abs(_recovered - _truth) > 1e-6 * (1.0 + abs(_truth)):
            raise _NotQuadraticError(
                "repr probe extraction did not reproduce the objective "
                f"(recovered {_recovered!r} vs true {_truth!r}); the probe "
                "differences cancelled — falling through to a stabler extractor"
            )

    # Extract constraints (same as LP)
    lp_data = _extract_lp_data_from_repr(model)
    n_slack = lp_data.c.shape[0] - n_orig

    if n_slack > 0:
        n_total = n_orig + n_slack
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            # Pad sparsely: densifying here would defeat the whole point (#863).
            Q_full = _sp.block_diag((Q, _sp.csr_matrix((n_slack, n_slack))), format="csr")
        else:
            Q_full = np.zeros((n_total, n_total), dtype=np.float64)
            Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Maximize → minimize -f: the QP backends always minimize, so negate the
    # whole quadratic (Q, c, constant). Without this the repr path returns the
    # raw maximize form — an indefinite Q for a concave-maximize objective — which
    # the QP solver rejects (and the autodiff fallback would have handled
    # correctly), silently yielding a wrong optimum. Mirrors the negation in
    # `_extract_lp_data_from_repr` and `_extract_qp_data_autodiff`. (Surfaced via
    # test_maximize_objective_sign_not_negated, issue #28.)
    if repr_.objective_sense == "maximize":
        Q_full = -Q_full
        c_full = -c_full
        d = -d

    return QPData(
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a
        # 0-d object array rather than raising, which would smuggle garbage into
        # every consumer (#863). Consumers densify via dense_Q().
        Q=Q_full if _sp_issparse(Q_full) else np.asarray(Q_full),  # type: ignore[arg-type]
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=d,
    )


def _extract_quadratic_coefficients_from_values(evaluate, n_vars: int):
    """Extract 0.5*x'Q*x + c'x + d from a quadratic scalar evaluator.

    Same probe identities as :func:`_extract_qp_data_from_repr`:
      - ``d = f(0)``
      - ``Q[j,j] = f(e_j) + f(-e_j) - 2d``
      - ``Q[i,j] = f(e_i+e_j) - f(e_i) - f(e_j) + d``  (``i != j``)
      - ``c_j = f(e_j) - d - 0.5*Q[j,j]``

    and now the same two #863 economies, which this (the QCP/QCQP probe path) was
    left out of because ``watercontamination0202`` does not route here — #868
    declined to widen speculatively, and #875 closes the gap rather than leave two
    extractors of one shape with two different cost profiles:

      * **Off-diagonal probing is restricted to the evaluator's support.** A variable
        absent from this row has ``f(e_j) == f(-e_j) == d`` and a zero diagonal, so
        every product involving it is identically zero. The support falls out of the
        ``O(n)`` diagonal probes already taken, so the restriction is free, and it
        takes the pair sweep from ``O(n^2)`` probes to ``O(|support|^2)``.
      * **Q is materialised through :func:`_materialise_Q`** — dense while ``(n, n)``
        float64 fits the budget (bit-identical to the ``np.zeros((n, n))`` it
        replaces, same entries written into the same zeros), scipy CSR beyond it. A
        dense ``(n, n)`` is 91 GB at ``n = 106,711``, and this function was called
        once per constraint.

    Both are pure cost reductions: the entries not probed are provably zero, and an
    entry absent from the accumulator densifies back to the 0.0 the dense array
    already held.
    """

    x_zero = np.zeros(n_vars, dtype=np.float64)
    d = float(evaluate(x_zero))

    f_ej = np.zeros(n_vars, dtype=np.float64)
    f_neg_ej = np.zeros(n_vars, dtype=np.float64)
    for j in range(n_vars):
        ej = np.zeros(n_vars, dtype=np.float64)
        ej[j] = 1.0
        f_ej[j] = float(evaluate(ej))
        ej[j] = -1.0
        f_neg_ej[j] = float(evaluate(ej))

    diag = f_ej + f_neg_ej - 2.0 * d

    terms: dict[tuple[int, int], float] = {}
    for j in range(n_vars):
        if diag[j] != 0.0:
            terms[(j, j)] = float(diag[j])

    support = [j for j in range(n_vars) if f_ej[j] != d or f_neg_ej[j] != d or diag[j] != 0.0]
    for _si, i in enumerate(support):
        for j in support[_si + 1 :]:
            eij = np.zeros(n_vars, dtype=np.float64)
            eij[i] = 1.0
            eij[j] = 1.0
            qij = float(evaluate(eij)) - f_ej[i] - f_ej[j] + d
            if qij != 0.0:
                terms[(i, j)] = qij
                terms[(j, i)] = qij

    Q = _materialise_Q(terms, n_vars)
    c_vec = f_ej - d - 0.5 * diag

    return Q, c_vec, d


def _extract_qcp_data_from_repr(model: Model) -> QCPData:
    """Extract QCP/QCQP data by evaluating the Rust ModelRepr."""

    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars
    Q, c_vec, obj_const = _extract_quadratic_coefficients_from_values(
        repr_.evaluate_objective,
        n_orig,
    )

    # COO accumulators rather than lists of dense rows (#863); see _materialise_A.
    ub_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    ub_rhs: list[float] = []
    eq_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    eq_rhs: list[float] = []
    q_rows: list[QuadraticConstraintData] = []

    for i in range(repr_.n_constraints):
        row_Q, row_c, row_const = _extract_quadratic_coefficients_from_values(
            lambda x, _i=i: repr_.evaluate_constraint(_i, x),
            n_orig,
        )
        sense = repr_.constraint_sense(i)
        rhs = float(repr_.constraint_rhs(i)) - float(row_const)
        if _quadratic_row_has_terms(row_Q):
            q_rows.append(
                QuadraticConstraintData(
                    # Preserve sparsity (see dense_Q / #863): np.asarray() on a scipy
                    # sparse matrix returns a 0-d object array instead of raising.
                    Q=row_Q if _sp_issparse(row_Q) else np.asarray(row_Q),  # type: ignore[arg-type]
                    c=np.asarray(row_c),  # type: ignore[arg-type]
                    sense=sense,
                    rhs=rhs,
                )
            )
            continue
        if sense == "==":
            _append_dense_row_coo(*eq_coo, len(eq_rhs), row_c)
            eq_rhs.append(rhs)
        elif sense == "<=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), row_c)
            ub_rhs.append(rhs)
        elif sense == ">=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), -row_c)
            ub_rhs.append(-rhs)

    x_l, x_u = _get_variable_bounds(model)
    A_ub = _materialise_A(*ub_coo, len(ub_rhs), n_orig)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)
    A_eq = _materialise_A(*eq_coo, len(eq_rhs), n_orig)
    b_eq = np.asarray(eq_rhs, dtype=np.float64)

    if repr_.objective_sense == "maximize":
        Q = -Q
        c_vec = -c_vec
        obj_const = -obj_const

    return QCPData(
        # Preserve sparsity (see dense_Q / #863, #875).
        Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
        c=np.asarray(c_vec),  # type: ignore[arg-type]
        # Preserve sparsity (see dense_A / #863).
        A_ub=A_ub if _sp_issparse(A_ub) else np.asarray(A_ub),  # type: ignore[arg-type]
        b_ub=np.asarray(b_ub),  # type: ignore[arg-type]
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        quadratic_constraints=tuple(q_rows),
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=float(obj_const),
    )


def extract_lp_data(model: Model) -> LPData:
    """Extract LP standard form from a model classified as LP.

    Tries Rust repr-based extraction first (for fast-API models), then
    algebraic extraction (for expression-based), then falls back to
    autodiff-based extraction if the DAG walk fails.

    Inequality constraints are converted to equalities with slacks:
      - body <= 0 becomes body + s = 0, s >= 0
      - body >= 0 becomes body - s = 0, s >= 0

    Args:
        model: A Model classified as ProblemClass.LP.

    Returns:
        LPData with c, A_eq, b_eq, x_l, x_u.
    """
    # Try repr-based extraction first (works for fast-API models)
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_lp_data_from_repr(model)
        except Exception as exc:  # noqa: BLE001 - falls through to the algebraic extractor
            # Each rung of this ladder is a *fast path*: a silent fall-through
            # turns "the repr extractor is slow" into an unexplained measurement.
            logger.debug("LP repr extraction (builder) failed: %s: %s", type(exc).__name__, exc)

    try:
        return extract_lp_data_algebraic(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the repr/autodiff extractors
        logger.debug("LP algebraic extraction failed: %s: %s", type(exc).__name__, exc)

    # Fast numeric repr probe before the expensive autodiff fallback. The Rust
    # ``ModelRepr`` evaluates fine without a ``_builder`` (same as
    # ``classify_problem``), so ``from_nl`` / repr-only models — where the
    # algebraic DAG walk can't run — skip the per-primitive eager-JAX autodiff
    # path (orders of magnitude faster on small instances; see #330).
    try:
        return _extract_lp_data_from_repr(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the autodiff extractor
        logger.debug(
            "LP repr extraction (probe) failed, falling back to autodiff: %s: %s",
            type(exc).__name__,
            exc,
        )

    return _extract_lp_data_autodiff(model)


def _extract_lp_data_autodiff(model: Model) -> LPData:
    """Extract LP standard form using autodiff (original slow path).

    Uses ``jax.jacobian`` rather than ``jax.grad`` so that vector-valued
    constraint bodies (DAE collocation residuals, MOL spatial residuals)
    are handled the same way as scalar bodies: each component contributes
    one row in the LP matrix, and inequalities get one slack per row.
    """
    import jax
    import jax.numpy as jnp

    from discopt._jax.dag_compiler import compile_constraint, compile_objective
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    # Extract c and constant: obj(x) = c'x + d, so grad(obj)(0) = c, obj(0) = d
    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)
    c = jax.grad(obj_fn)(x_zero)
    obj_const = float(obj_fn(x_zero))

    # Extract constraint coefficients
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # First pass: compile each constraint and probe its row count by
    # evaluating at zero. Scalar bodies become a single row; vector bodies
    # contribute one row per component.
    eq_blocks: list[tuple[jnp.ndarray, jnp.ndarray]] = []  # (J, body0)
    ineq_blocks: list[tuple[jnp.ndarray, jnp.ndarray, str]] = []  # (J, body0, sense)

    for con in constraints:
        con_fn = compile_constraint(con, model)
        # Flatten any vector / matrix constraint body into a length-k vector;
        # each component becomes its own LP row.
        body0 = jnp.asarray(con_fn(x_zero), dtype=jnp.float64).reshape(-1)
        jac_raw = jax.jacobian(lambda x, _f=con_fn: jnp.asarray(_f(x)).reshape(-1))(x_zero)
        jac = jnp.asarray(jac_raw, dtype=jnp.float64).reshape(body0.shape[0], n_orig)
        if con.sense == "==":
            eq_blocks.append((jac, body0))
        elif con.sense == "<=":
            ineq_blocks.append((jac, body0, "le"))
        elif con.sense == ">=":
            ineq_blocks.append((jac, body0, "ge"))

    n_eq_rows = sum(int(j.shape[0]) for j, _ in eq_blocks)
    n_ineq_rows = sum(int(j.shape[0]) for j, _, _ in ineq_blocks)
    n_slack = n_ineq_rows
    n_total = n_orig + n_slack

    A_rows: list[jnp.ndarray] = []
    b_vals: list[float] = []

    for jac, body0 in eq_blocks:
        for r in range(jac.shape[0]):
            A_rows.append(jnp.concatenate([jac[r], jnp.zeros(n_slack)]))
            b_vals.append(-float(body0[r]))

    slack_offset = 0
    for jac, body0, sense in ineq_blocks:
        for r in range(jac.shape[0]):
            slack_col = jnp.zeros(n_slack)
            sign = 1.0 if sense == "le" else -1.0
            # body ≤ 0 → body + s = 0, s ≥ 0; body ≥ 0 → body − s = 0, s ≥ 0.
            slack_col = slack_col.at[slack_offset].set(sign)
            A_rows.append(jnp.concatenate([jac[r], slack_col]))
            b_vals.append(-float(body0[r]))
            slack_offset += 1

    m_total = n_eq_rows + n_ineq_rows
    if m_total > 0:
        A_eq = jnp.stack(A_rows)
        b_eq = jnp.array(b_vals, dtype=jnp.float64)
    else:
        A_eq = jnp.zeros((0, n_total), dtype=jnp.float64)
        b_eq = jnp.zeros(0, dtype=jnp.float64)

    # Bounds: original vars keep their bounds, slack vars >= 0
    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = jnp.concatenate([c, jnp.zeros(n_slack)])
    x_l = jnp.concatenate([x_l_orig, jnp.zeros(n_slack)])
    x_u = jnp.concatenate([x_u_orig, jnp.full(n_slack, jnp.inf)])

    # Handle objective sense: negate for maximization (solvers always minimize).
    # C-30: this autodiff fallback previously dropped the maximize negation that
    # every other extractor applies (extract_lp_data_algebraic:743,
    # _extract_lp_data_from_repr:927, _extract_qp_data_autodiff:1375), so a
    # `maximize` model routed here (e.g. a vector `sum(const*var)` body that the
    # algebraic walk refuses) was silently minimized and returned 0.
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        c=c_full,
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data(model: Model) -> QPData:
    """Extract QP standard form from a model classified as QP.

    Tries Rust repr-based extraction first (for fast-API models), then
    algebraic extraction (for expression-based), then falls back to
    autodiff-based extraction if the DAG walk fails.

    Args:
        model: A Model classified as ProblemClass.QP.

    Returns:
        QPData with Q, c, A_eq, b_eq, x_l, x_u.
    """
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_qp_data_from_repr(model)
        except Exception as exc:  # noqa: BLE001 - falls through to the algebraic extractor
            # See the LP ladder above: a silent fall-through here is how a fast
            # path disappears without any evidence that it did.
            logger.debug("QP repr extraction (builder) failed: %s: %s", type(exc).__name__, exc)

    try:
        return extract_qp_data_algebraic(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the repr/autodiff extractors
        logger.debug("QP algebraic extraction failed: %s: %s", type(exc).__name__, exc)

    # Fast numeric repr probe before the expensive autodiff fallback. The Rust
    # ``ModelRepr`` evaluates fine without a ``_builder`` (same as
    # ``classify_problem``), so ``from_nl`` / repr-only models — where the
    # algebraic DAG walk can't run — skip the per-primitive eager-JAX autodiff
    # path (orders of magnitude faster on small instances; see #330).
    try:
        return _extract_qp_data_from_repr(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the autodiff extractor
        logger.debug(
            "QP repr extraction (probe) failed, falling back to autodiff: %s: %s",
            type(exc).__name__,
            exc,
        )

    return _extract_qp_data_autodiff(model)


def extract_qcp_data(model: Model) -> QCPData:
    """Extract QCP/QCQP data from a model classified as QCP/QCQP/MIQCP/MIQCQP."""
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_qcp_data_from_repr(model)
        except Exception as exc:  # noqa: BLE001 - falls through to the algebraic extractor
            logger.debug("QCP repr extraction failed: %s: %s", type(exc).__name__, exc)

    return extract_qcp_data_algebraic(model)


def _extract_qp_data_autodiff(model: Model) -> QPData:
    """Extract QP standard form using autodiff (original slow path)."""
    import jax
    import jax.numpy as jnp

    from discopt._jax.dag_compiler import compile_objective
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)

    # Q = hessian(obj) — constant for QP
    Q = jax.hessian(obj_fn)(x_zero)

    # c = grad(obj)(0) = Q*0 + c = c (linear part)
    c_vec = jax.grad(obj_fn)(x_zero)

    # Constant term: f(0) = 0.5*0'Q*0 + c'*0 + d = d
    obj_const = float(obj_fn(x_zero))

    # Extract LP data for constraints (they're all linear)
    lp_data = extract_lp_data(model)
    n_slack = lp_data.c.shape[0] - n_orig

    # Extend Q with zeros for slack variables
    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = jnp.zeros((n_total, n_total), dtype=jnp.float64)
        Q_full = Q_full.at[:n_orig, :n_orig].set(Q)
        c_full = jnp.concatenate([c_vec, jnp.zeros(n_slack)])
    else:
        Q_full = Q
        c_full = c_vec

    # Handle objective sense: negate for maximization (solvers always minimize)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q_full = -Q_full
        c_full = -c_full
        obj_const = -obj_const

    return QPData(
        Q=Q_full,
        c=c_full,
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=obj_const,
    )
